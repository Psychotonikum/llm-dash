"""Global stats from LiteLLM Postgres (LiteLLM_SpendLogs).

Every local and online request flows through the gateway, so this table is the
complete source for 1h/1d/1M/all request + token totals, split LOCAL vs ONLINE
by api_base (homelab LAN = local GPU model; anything else = online API).

Reads are over the read-only `dashro` role via the SSH tunnel (127.0.0.1:15432).
The whole table is small (thousands of rows) so we cache it and refresh
periodically; bucketing happens in memory.
"""
import asyncio
import datetime as dt
import asyncpg

RANGE_SECONDS = {"1h": 3600, "1d": 86400, "1M": 30 * 86400, "all": None}
BUCKET_SECONDS = {"1h": 300, "1d": 3600, "1M": 86400, "all": 86400}
FAIL_STATUSES = {"error", "failed", "exception", "timeout"}


class GlobalStats:
    def __init__(self, cfg, refresh_interval=60):
        self.cfg = cfg
        self.refresh_interval = refresh_interval
        self._rows = []
        self._last_refresh = 0.0
        self._pool = None
        self._lock = asyncio.Lock()
        self._ok = False
        self._err = None
        self._local_model_set = {
            "".join(ch for ch in p.lower() if ch.isalnum())
            for p in cfg.local_models
        }

    async def _pool_or_err(self):
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    host=self.cfg.pg_host,
                    port=self.cfg.pg_port,
                    user=self.cfg.pg_user,
                    password=self.cfg.pg_password,
                    database=self.cfg.pg_db,
                    min_size=1, max_size=3,
                )
                self._ok = True
            except Exception as e:
                self._err = "pg connect: %s" % e
                return None
        return self._pool

    async def _refresh(self, force=False):
        import time
        if not force and (time.time() - self._last_refresh) < self.refresh_interval:
            return
        pool = await self._pool_or_err()
        if pool is None:
            return
        async with pool.acquire() as d:
            rows = await d.fetch(
                'SELECT "startTime" AS st, model, "api_base", '
                'prompt_tokens AS pt, completion_tokens AS ct, status '
                'FROM "LiteLLM_SpendLogs" ORDER BY "startTime"'
            )
        def _aware(st):
            if st is not None and st.tzinfo is None:
                st = st.replace(tzinfo=dt.timezone.utc)
            return st
        self._rows = [(_aware(r["st"]), r["model"], r["api_base"], r["pt"],
                       r["ct"], (r["status"] or "").lower()) for r in rows]
        import time as _t
        self._last_refresh = _t.time()
        self._err = None

    def _is_local(self, api_base, model=None):
        # Explicit local model names win: spend-log api_base is often empty,
        # which would otherwise classify a local model as online.
        if model:
            norm = "".join(ch for ch in model.lower() if ch.isalnum())
            if norm in self._local_model_set:
                return True
        if not api_base:
            return False
        return any(api_base.startswith(p) for p in self.cfg.local_prefixes)

    async def query(self, rng="1d"):
        rng = rng if rng in RANGE_SECONDS else "1d"
        async with self._lock:
            await self._refresh()
        if self._pool is None:
            return {"ok": False, "error": self._err, "range": rng}

        now = dt.datetime.now(dt.timezone.utc)
        span = RANGE_SECONDS[rng]
        since = (now - dt.timedelta(seconds=span)) if span else None
        bsec = BUCKET_SECONDS[rng]

        rows = self._rows
        # determine bucket range for a continuous axis
        first_st = rows[0][0] if rows and rows[0][0] else now
        start_bucket = (since or first_st)
        sb = int(start_bucket.timestamp() // bsec) * bsec
        eb = int(now.timestamp() // bsec) * bsec

        buckets = {}
        local_models = {}
        online_models = {}
        for st, model, api_base, pt, ct, status in rows:
            if since and st < since:
                continue
            local = self._is_local(api_base, model)
            if model:
                mc = local_models if local else online_models
                mc[model] = mc.get(model, 0) + 1
            key = int(st.timestamp() // bsec) * bsec
            b = buckets.setdefault(key, dict(
                local_req=0, online_req=0,
                local_in=0, local_out=0, online_in=0, online_out=0,
                local_fail=0, online_fail=0))
            if local:
                b["local_req"] += 1
                b["local_in"] += pt or 0
                b["local_out"] += ct or 0
                if status in FAIL_STATUSES:
                    b["local_fail"] += 1
            else:
                b["online_req"] += 1
                b["online_in"] += pt or 0
                b["online_out"] += ct or 0
                if status in FAIL_STATUSES:
                    b["online_fail"] += 1

        out = []
        for k in range(sb, eb + 1, bsec):
            b = buckets.get(k, dict(
                local_req=0, online_req=0,
                local_in=0, local_out=0, online_in=0, online_out=0,
                local_fail=0, online_fail=0))
            out.append({
                "ts": dt.datetime.fromtimestamp(k, dt.timezone.utc).isoformat(),
                "local_req": b["local_req"], "online_req": b["online_req"],
                "req": b["local_req"] + b["online_req"],
                "local_in": b["local_in"], "local_out": b["local_out"],
                "online_in": b["online_in"], "online_out": b["online_out"],
                "local_tok": b["local_in"] + b["local_out"],
                "online_tok": b["online_in"] + b["online_out"],
                "tok": (b["local_in"] + b["local_out"]) + (b["online_in"] + b["online_out"]),
                "fail": b["local_fail"] + b["online_fail"],
            })

        tot = {
            "req": sum(b["req"] for b in out),
            "tok": sum(b["tok"] for b in out),
            "local_req": sum(b["local_req"] for b in out),
            "online_req": sum(b["online_req"] for b in out),
            "local_tok": sum(b["local_tok"] for b in out),
            "online_tok": sum(b["online_tok"] for b in out),
            "local_in": sum(b["local_in"] for b in out),
            "local_out": sum(b["local_out"] for b in out),
            "online_in": sum(b["online_in"] for b in out),
            "online_out": sum(b["online_out"] for b in out),
            "fail": sum(b["fail"] for b in out),
        }
        tot["local_pct"] = round(100.0 * tot["local_tok"] / tot["tok"], 1) if tot["tok"] else 0.0
        tot["local_req_pct"] = round(100.0 * tot["local_req"] / tot["req"], 1) if tot["req"] else 0.0
        result = {
            "ok": True, "range": rng, "buckets": out, "totals": tot,
            "rows": len(rows),
            "local_models": dict(sorted(local_models.items(), key=lambda x: -x[1])[:8]),
            "online_models": dict(sorted(online_models.items(), key=lambda x: -x[1])[:8]),
        }
        if rng not in ("1h", "1d"):
            days = {}
            for st, model, api_base, pt, ct, status in rows:
                if since and st < since:
                    continue
                local = self._is_local(api_base, model)
                dkey = st.astimezone(dt.timezone.utc).strftime("%Y-%m-%d")
                dd = days.setdefault(dkey, dict(local_req=0, online_req=0,
                                               local_tok=0, online_tok=0))
                tok = (pt or 0) + (ct or 0)
                if local:
                    dd["local_req"] += 1
                    dd["local_tok"] += tok
                else:
                    dd["online_req"] += 1
                    dd["online_tok"] += tok
            result["days"] = [
                {"date": k, "req": v["local_req"] + v["online_req"],
                 "tok": v["local_tok"] + v["online_tok"], **v}
                for k, v in sorted(days.items())
            ]
            result["span_days"] = (now - first_st).days
        return result

    async def close(self):
        if self._pool:
            await self._pool.close()
