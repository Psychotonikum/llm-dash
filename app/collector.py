"""Node collector: polls llama-swap nodes, tracks reachability and the live
request, derives the GPU/inference state.

Defensive across llama-swap versions (v241 and v265 expose different /api
surfaces) - every endpoint is optional; only /v1/models gates "online".

Live-request + state source:
  /v1/models           -> configured models + loaded/unloaded status
  /running             -> live processes (model, state: ready/loading/...)
  /api/metrics/activity-> recent requests incl. live tok/s, prompt/s, duration
  /api/metrics/stats   -> totals + prompt/gen latency p50/p95/p99
  /api/performance     -> GPU temp/util/VRAM/power/fan (ring of samples)
  /api/version         -> build/version

State derivation (per node):
  offline  -> core endpoint unreachable
  prefill  -> active request, high prompt_per_second, ~no output tokens yet
  generating-> active request, output tokens streaming (tokens_per_second>0)
  processing-> active request, neither clearly prefill nor generating
  loading  -> a model process is starting
  idle     -> model ready, no active request
  unloaded -> no model running
"""
import asyncio
import time
from datetime import datetime, timezone
from urllib.parse import quote
import httpx


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class NodeState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg["name"]
        self.online = False
        self.last_seen = None          # last successful poll (iso)
        self.last_error = None
        # last good snapshot (kept while offline); seeded with identity so a
        # never-online node still renders a full card
        self.snapshot = {
            "name": cfg["name"], "label": cfg.get("label", cfg["name"]),
            "ip": cfg["ip"], "endpoint": cfg["base"], "base": cfg["base"],
            "online": False, "state": "offline", "models": [], "gpu": {},
            "stats": {}, "current": None, "activity": [], "running": [],
            "version": None, "now": None, "last_error": None, "last_seen": None,
        }
        self.prev_top = None           # (id, duration_ms) of top activity entry
        self.version = None
        self.poll_count = 0
        self.fail_count = 0
        self.busy_since = None         # ts GPU crossed busy threshold (request in flight)


class NodeCollector:
    # GPU util % that counts as "a request is in flight". Idle llama-server
    # sits at ~0%; prefill/generation run at 60-100%.
    BUSY_UTIL = 5.0

    def __init__(self, cfg, client):
        self.cfg = cfg
        self.client = client
        self.b = cfg["base"]
        self.name = cfg["name"]
        self.state = NodeState(cfg)

    async def _get(self, path):
        try:
            r = await self.client.get(self.b + path)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def _is_live(self, top):
        """True only if the top activity entry is a request still in flight.

        llama-swap only appends to /api/metrics/activity when a request
        COMPLETES, so a new id appearing is a finished request, not a live
        one. The only unambiguous live signal is the same id's duration_ms
        growing between polls. On the first observation we cannot tell —
        return False there; the GPU-utilization busy-streak (see poll())
        covers the in-flight state regardless.
        """
        st = self.state
        if not top:
            st.prev_top = None
            return False
        tid = top.get("id")
        tdur = top.get("duration_ms") or 0
        prev = st.prev_top
        st.prev_top = (tid, tdur)
        if prev is None:
            return False
        pid, pdur = prev
        return tid == pid and tdur > pdur

    async def poll(self):
        st = self.state
        (models_d, act_d, stats_d, perf_d, run_d, ver_d) = await asyncio.gather(
            self._get("/v1/models"),
            self._get("/api/metrics/activity"),
            self._get("/api/metrics/stats"),
            self._get("/api/performance"),
            self._get("/running"),
            self._get("/api/version"),
        )
        st.poll_count += 1
        online = isinstance(models_d, dict)
        if not online and (isinstance(run_d, dict) or isinstance(ver_d, dict)):
            online = True

        if online:
            st.online = True
            st.last_seen = _now_iso()
            st.last_error = None
            st.fail_count = 0
            if isinstance(ver_d, dict):
                st.version = ver_d.get("version")
        else:
            st.online = False
            st.fail_count += 1
            st.last_error = st.last_error or "node unreachable"
            st.busy_since = None

        if not online:
            # keep the last good snapshot, just flag it stale
            st.snapshot["online"] = False
            st.snapshot["state"] = "offline"
            st.snapshot["last_error"] = st.last_error
            st.snapshot["now"] = _now_iso()
            return st.snapshot

        # ---- build snapshot from live data ----
        models = []
        if isinstance(models_d, dict):
            for m in models_d.get("data", []):
                models.append({
                    "id": m.get("id"),
                    "name": m.get("name"),
                    "status": (m.get("status") or {}).get("value", "unknown"),
                    "context_length": m.get("context_length"),
                })

        running, loading = [], []
        if isinstance(run_d, dict):
            for r in run_d.get("running", []):
                entry = {"model": r.get("model"), "state": r.get("state")}
                running.append(entry)
                if r.get("state") in ("loading", "starting", "warming"):
                    loading.append(entry)

        activity = act_d.get("data", []) if isinstance(act_d, dict) else []
        # keep only the most recent handful to bound size
        activity = activity[:8]

        top = activity[0] if activity else None
        live_top = self._is_live(top)

        current = None
        if live_top and top:
            t = top.get("tokens") or {}
            current = {
                "id": top.get("id"),
                "model": top.get("model"),
                "req_path": top.get("req_path"),
                "status": top.get("resp_status_code"),
                "duration_ms": top.get("duration_ms"),
                "elapsed_s": round((top.get("duration_ms") or 0) / 1000, 1),
                "input_tokens": t.get("input_tokens"),
                "output_tokens": t.get("output_tokens"),
                "cache_tokens": t.get("cache_tokens"),
                "tokens_per_second": round(t.get("tokens_per_second") or 0, 2),
                "prompt_per_second": round(t.get("prompt_per_second") or 0, 1),
                "streaming": str(top.get("resp_content_type", ""))
                .startswith("text/event-stream"),
            }

        stats = None
        if isinstance(stats_d, dict):
            ph = stats_d.get("prompt_histogram") or {}
            gh = stats_d.get("gen_histogram") or {}
            stats = {
                "total_requests": stats_d.get("total_requests"),
                "total_input_tokens": stats_d.get("total_input_tokens"),
                "total_output_tokens": stats_d.get("total_output_tokens"),
                "total_cache_tokens": stats_d.get("total_cache_tokens"),
                "prompt_p50": _r(ph.get("p50")), "prompt_p95": _r(ph.get("p95")),
                "prompt_p99": _r(ph.get("p99")),
                "gen_p50": _r(gh.get("p50")), "gen_p95": _r(gh.get("p95")),
                "gen_p99": _r(gh.get("p99")),
            }

        gpu = None
        if isinstance(perf_d, dict):
            samples = perf_d.get("gpu_stats") or []
            if samples:
                g = samples[-1]
                mu = g.get("mem_util_pct")
                mu = _r(mu) if isinstance(mu, (int, float)) else None
                if mu is None:
                    used, total = g.get("mem_used_mb"), g.get("mem_total_mb")
                    if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total:
                        mu = _r(100.0 * used / total)
                used, total = g.get("mem_used_mb"), g.get("mem_total_mb")
                free = (total - used) if (
                    isinstance(used, (int, float)) and isinstance(total, (int, float))) else None
                gpu = {
                    "name": g.get("name"),
                    "temp_c": g.get("temp_c"),
                    "gpu_util_pct": g.get("gpu_util_pct"),
                    "mem_used_mb": used,
                    "mem_total_mb": total,
                    "mem_used_pct": mu,          # canonical key (frontend)
                    "mem_util_pct": mu,          # alias, kept for compat
                    "mem_free_mb": free,
                    "power_draw_w": _r(g.get("power_draw_w")),
                    "fan_speed_pct": g.get("fan_speed_pct"),
                }

        # llama-swap (v241/v265) does not list in-flight requests in
        # /api/metrics/activity — only completed ones. An active request shows
        # up as GPU utilization on the performance sample, so track a busy
        # streak: it yields the "generating" state plus an in-progress
        # "current request" (model + elapsed time) for the UI.
        loaded = [m for m in models if m.get("status") == "loaded"]
        gpu_util = (gpu or {}).get("gpu_util_pct")
        has_model = bool(loaded) or bool(running)
        busy = (
            isinstance(gpu_util, (int, float))
            and gpu_util >= self.BUSY_UTIL
            and has_model
        )
        if busy:
            if st.busy_since is None:
                st.busy_since = time.time()
        else:
            st.busy_since = None
        if current is None and busy:
            mname = (top.get("model") if top else None) or (
                loaded[0].get("id") if loaded else
                (running[0].get("model") if running else None)
            )
            if mname:
                current = {
                    "model": mname,
                    "in_progress": True,
                    "elapsed_s": round(time.time() - st.busy_since, 1)
                    if st.busy_since else None,
                    "gpu_util_pct": _r(gpu_util),
                }

        snap = {
            "name": self.name,
            "label": self.cfg.get("label"),
            "ip": self.cfg["ip"],
            "port": self.cfg.get("port", 8080),
            "base": self.b,
            "endpoint": self.b,
            "online": True,
            "last_seen": st.last_seen,
            "last_error": None,
            "now": _now_iso(),
            "version": st.version,
            "models": models,
            "running": running,
            "current": current,
            "activity": activity,
            "stats": stats,
            "gpu": gpu,
            "state": self._derive_state(models, running, loading, current),
        }
        st.snapshot = snap
        return snap

    @staticmethod
    def _derive_state(models, running, loading, current):
        if loading:
            return "loading"
        if current:
            if current.get("in_progress"):
                return "generating"
            tps = current.get("tokens_per_second") or 0
            pps = current.get("prompt_per_second") or 0
            out = current.get("output_tokens") or 0
            if out > 0 and tps > 0.5:
                return "generating"
            if pps > 5:
                return "prefill"
            return "processing"
        if running:
            return "idle"
        loaded = [m for m in models if m.get("status") == "loaded"]
        if loaded:
            return "idle"
        return "unloaded"

    # ---- actions (schedule enforcement + manual card actions) ----
    async def load_model(self, model):
        try:
            r = await self.client.get(self.b + "/upstream/" + quote(model, safe=""),
                                      timeout=900)
        except Exception as e:
            return {"ok": False, "error": f"node unreachable ({type(e).__name__})"}
        return _action_result(r)

    async def unload_model(self, model):
        try:
            r = await self.client.post(
                self.b + "/api/models/unload/" + quote(model, safe=""), timeout=120)
        except Exception as e:
            return {"ok": False, "error": f"node unreachable ({type(e).__name__})"}
        return _action_result(r)

    async def unload_all(self):
        try:
            r = await self.client.post(self.b + "/api/models/unload", timeout=120)
        except Exception as e:
            return {"ok": False, "error": f"node unreachable ({type(e).__name__})"}
        return _action_result(r)


def _r(v):
    return round(v, 1) if isinstance(v, (int, float)) else v


def _action_result(r):
    try:
        body = r.json()
    except Exception:
        body = r.text[:300]
    return {"status": r.status_code, "ok": r.status_code == 200, "body": body}


class CollectorManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nodes = {}
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.node_timeout, connect=cfg.node_timeout),
            limits=httpx.Limits(max_connections=60),
            headers={"Accept": "application/json"},
        )
        for n in cfg.nodes:
            self.nodes[n["name"]] = NodeCollector(n, self.client)

    def all(self):
        return [self.nodes[n].state.snapshot for n in self.cfg.node_names]

    # ---- runtime node management (web UI add/edit/delete) ----
    @staticmethod
    def _entry_with_base(entry):
        entry = dict(entry)
        entry["base"] = "http://%s:%s" % (entry["ip"], entry.get("port", 8080))
        return entry

    def add_node(self, entry):
        entry = self._entry_with_base(entry)
        col = NodeCollector(entry, self.client)
        self.nodes[entry["name"]] = col
        self.cfg.nodes.append(entry)
        self.cfg.node_names.append(entry["name"])
        return col

    def update_node(self, name, entry):
        if name not in self.nodes:
            raise KeyError(name)
        col = self.nodes.pop(name)
        entry = self._entry_with_base(entry)
        col.cfg = entry
        col.b = entry["base"]
        col.name = entry["name"]
        col.state.name = entry["name"]
        # refresh snapshot identity fields (seeded once at NodeState init)
        for k in ("name", "label", "ip", "base"):
            col.state.snapshot[k] = entry.get(k, col.state.snapshot.get(k))
        col.state.snapshot["port"] = entry.get("port", 8080)
        self.nodes[entry["name"]] = col
        self.cfg.nodes = [e for e in self.cfg.nodes if e.get("name") != name]
        self.cfg.nodes.append(entry)
        self.cfg.node_names = [n for n in self.cfg.node_names if n != name]
        self.cfg.node_names.append(entry["name"])
        return col

    def remove_node(self, name):
        self.nodes.pop(name, None)
        self.cfg.nodes = [e for e in self.cfg.nodes if e.get("name") != name]
        self.cfg.node_names = [n for n in self.cfg.node_names if n != name]

    async def poll_all(self):
        results = await asyncio.gather(
            *[c.poll() for c in self.nodes.values()], return_exceptions=True)
        out = []
        for name, r in zip(self.cfg.node_names, results):
            if isinstance(r, Exception):
                st = self.nodes[name].state
                st.online = False
                st.last_error = str(r)
                out.append(st.snapshot)
            else:
                out.append(r)
        return out

    async def load_model(self, node, model):
        return await self.nodes[node].load_model(model)

    async def unload_model(self, node, model):
        return await self.nodes[node].unload_model(model)

    async def unload_all(self, node):
        return await self.nodes[node].unload_all()

    async def close(self):
        await self.client.aclose()
