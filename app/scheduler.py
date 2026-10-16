"""Per-model schedules (on/off windows, per weekday) enforced by llama-swap
model load/unload.

- ON  window: model should be loaded (warm) on its node.
- OFF window: model should be unloaded (frees VRAM).

Enforcement only acts on user-created schedules and only on state transitions.
Model load is slow (minutes) so it runs in a background task with an in-flight
guard to avoid double-triggering. SCHEDULE_ENFORCE=0 => preview only (log, don't
act).
"""
import asyncio
import datetime as dt


class Scheduler:
    def __init__(self, cfg, db, collectors):
        self.cfg = cfg
        self.db = db
        self.collectors = collectors
        self.in_flight = set()          # (node, model) currently loading/unloading
        self.log = []                   # recent enforcement events (ring)

    # ---------- window math ----------
    @staticmethod
    def _to_min(hhmm):
        try:
            h, m = hhmm.split(":")
            return int(h) * 60 + int(m)
        except Exception:
            return 0

    def _in_window(self, on_time, off_time, now):
        on, off, cur = (self._to_min(on_time), self._to_min(off_time),
                        now.hour * 60 + now.minute)
        if on <= off:
            return on <= cur <= off
        return cur >= on or cur <= off            # wraps midnight

    def window_active(self, s, now=None):
        now = now or dt.datetime.now()
        if not s.get("enabled", False):
            return False
        days = s.get("days") or "1111111"
        wd = now.weekday()
        if wd >= len(days) or days[wd] != "1":
            return False
        return self._in_window(s.get("on_time", "00:00"),
                               s.get("off_time", "23:59"), now)

    def next_transition(self, s, now=None):
        now = now or dt.datetime.now()

        def active_at(t):
            if not s.get("enabled", False):
                return False
            days = s.get("days") or "1111111"
            if t.weekday() >= len(days) or days[t.weekday()] != "1":
                return False
            return self._in_window(s.get("on_time", "00:00"),
                                   s.get("off_time", "23:59"), t)

        cur = active_at(now)
        base = now.replace(hour=0, minute=0, second=0, microsecond=0)
        cands = []
        for d in range(0, 15):
            day = base + dt.timedelta(days=d)
            for hhmm in (s.get("on_time", "00:00"), s.get("off_time", "23:59")):
                try:
                    h, m = hhmm.split(":")
                    cands.append(day.replace(hour=int(h), minute=int(m)))
                except Exception:
                    pass
            cands.append(day)                     # midnight (mask change)
        cands = sorted(c for c in cands if c > now)
        for t in cands:
            if active_at(t) != cur:
                return t
        return None

    # ---------- state ----------
    def _is_loaded(self, snap, model):
        if not snap:
            return False
        for r in snap.get("running", []):
            if r.get("model") == model and r.get("state") in (
                    "ready", "loaded", "running", "warming"):
                return True
        for m in snap.get("models", []):
            if m.get("id") == model and m.get("status") == "loaded":
                return True
        return False

    def _log(self, action, node, model, res):
        import datetime
        self.log.append({
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action": action, "node": node, "model": model,
            "status": (res or {}).get("status") if isinstance(res, dict) else None,
            "ok": (res or {}).get("ok") if isinstance(res, dict) else None,
        })
        if len(self.log) > 60:
            self.log = self.log[-60:]

    async def _do(self, col, action, node, model, s):
        key = (node, model)
        try:
            if action == "load":
                res = await col.load_model(model)
            else:
                res = await col.unload_model(model)
            self._log(action, node, model, res)
            self.db.set_state(s["id"], 1 if action == "load" else 0,
                              "%s:%s" % (action, (res or {}).get("status")))
        except Exception as e:
            self._log(action + ":error", node, model, {"ok": False, "status": str(e)})
        finally:
            self.in_flight.discard(key)

    async def enforce_once(self):
        now = dt.datetime.now()
        for s in self.db.list():
            if not s.get("enabled"):
                continue
            col = self.collectors.nodes.get(s["node"])
            if col is None:
                continue
            if not col.state.online:
                continue                       # don't act on an offline node
            model = s["model"]
            key = (s["node"], model)
            if key in self.in_flight:
                continue
            want = self.window_active(s, now)
            loaded = self._is_loaded(col.state.snapshot, model)
            if loaded == want:
                if s.get("last_state") != (1 if want else 0):
                    self.db.set_state(s["id"], 1 if want else 0, "steady")
                continue
            if not self.cfg.schedule_enforce:
                self._log("preview:" + ("load" if want else "unload"),
                          s["node"], model, None)
                self.db.set_state(s["id"], 1 if want else 0, "preview")
                continue
            self.in_flight.add(key)
            action = "load" if want else "unload"
            asyncio.create_task(self._do(col, action, s["node"], model, s))

    async def run(self, stop):
        while not stop.is_set():
            try:
                await self.enforce_once()
            except Exception as e:
                self._log("tick:error", "-", "-", {"ok": False, "status": str(e)})
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.cfg.schedule_interval)
            except asyncio.TimeoutError:
                pass
