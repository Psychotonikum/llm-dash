"""llamaDash FastAPI app: background poller + scheduler, JSON API, static UI."""
import asyncio
import datetime as dt
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import CFG
from .db import DB
from .collector import CollectorManager
from .globalstats import GlobalStats
from .scheduler import Scheduler

STATE = {}
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = DB(CFG.data_dir / "schedules.db")
    collectors = CollectorManager(CFG)
    stats = GlobalStats(CFG)
    scheduler = Scheduler(CFG, db, collectors)
    STATE.update(db=db, collectors=collectors, stats=stats, sched=scheduler)
    stop = asyncio.Event()

    async def poller():
        while not stop.is_set():
            try:
                await collectors.poll_all()
            except Exception as e:
                print("poll error:", e)
            try:
                await asyncio.wait_for(stop.wait(), timeout=CFG.poll_interval)
            except asyncio.TimeoutError:
                pass

    p = asyncio.create_task(poller())
    s = asyncio.create_task(scheduler.run(stop))
    # initial poll so the UI has data immediately
    await collectors.poll_all()
    yield
    stop.set()
    p.cancel(); s.cancel()
    await collectors.close()
    await stats.close()


app = FastAPI(title="llamaDash", lifespan=lifespan)


def _now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


# ---------------- health / nodes ----------------
@app.get("/api/health")
async def health():
    cols = STATE.get("collectors")
    nodes = cols.all() if cols else []
    return {
        "ok": True,
        "ts": _now_iso(),
        "port": CFG.port,
        "nodes_total": len(nodes),
        "nodes_online": sum(1 for n in nodes if n.get("online")),
        "schedule_enforce": CFG.schedule_enforce,
    }


@app.get("/api/nodes")
async def nodes():
    cols = STATE["collectors"]
    return {"ts": _now_iso(), "nodes": cols.all()}


@app.get("/api/nodes/{name}")
async def node(name: str):
    cols = STATE["collectors"]
    if name not in cols.nodes:
        raise HTTPException(404, "unknown node")
    return cols.nodes[name].state.snapshot


# ---------------- node management (add/edit/delete) ----------------
NODE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


class NodeIn(BaseModel):
    name: str
    label: str = ""
    ip: str
    port: int = 8080


def _validate_node(body: "NodeIn", existing, rename_from=None):
    name = (body.name or "").strip()
    if not NODE_NAME_RE.match(name):
        raise HTTPException(400, "name: a-z 0-9 - _ only, start alnum, max 32")
    ip = (body.ip or "").strip()
    if not IP_RE.match(ip) or any(int(o) > 255 for o in ip.split(".")):
        raise HTTPException(400, "ip: invalid IPv4 address")
    if not (1 <= body.port <= 65535):
        raise HTTPException(400, "port: 1-65535")
    if name in existing and name != rename_from:
        raise HTTPException(400, "node '%s' already exists" % name)
    body.name, body.ip = name, ip


@app.post("/api/nodes")
async def add_node(body: NodeIn):
    cols = STATE["collectors"]
    _validate_node(body, cols.cfg.node_names)
    entry = {"name": body.name, "label": body.label or body.name,
             "ip": body.ip, "port": body.port}
    col = cols.add_node(entry)
    from .config import save_nodes
    save_nodes(cols.cfg.nodes)
    await col.poll()  # immediate first snapshot
    return col.state.snapshot


@app.put("/api/nodes/{name}")
async def update_node(name: str, body: NodeIn):
    cols = STATE["collectors"]
    if name not in cols.nodes:
        raise HTTPException(404, "unknown node")
    _validate_node(body, cols.cfg.node_names, rename_from=name)
    entry = {"name": body.name, "label": body.label or body.name,
             "ip": body.ip, "port": body.port}
    col = cols.update_node(name, entry)
    from .config import save_nodes
    save_nodes(cols.cfg.nodes)
    await col.poll()  # immediate re-snapshot under the new identity
    return col.state.snapshot


@app.delete("/api/nodes/{name}")
async def delete_node(name: str):
    cols = STATE["collectors"]
    if name not in cols.nodes:
        raise HTTPException(404, "unknown node")
    orphaned = [s["id"] for s in STATE["db"].list() if s["node"] == name]
    cols.remove_node(name)
    from .config import save_nodes
    save_nodes(cols.cfg.nodes)
    return {"ok": True, "orphaned_schedules": orphaned}


# ---------------- global stats ----------------
@app.get("/api/global")
async def global_stats(range: str = "1d"):
    return await STATE["stats"].query(range)


# ---------------- schedules ----------------
class SchedIn(BaseModel):
    model: str
    node: str
    days: str = "1111111"
    on_time: str = "00:00"
    off_time: str = "23:59"
    enabled: bool = True
    note: str = ""


def _sched_view(s: dict):
    sch = STATE["sched"]
    nt = sch.next_transition(s)
    return {**s,
            "active_now": sch.window_active(s),
            "next_transition": nt.isoformat() if nt else None}


@app.get("/api/schedules")
async def list_schedules():
    return {"ts": _now_iso(),
            "enforce": CFG.schedule_enforce,
            "schedules": [_sched_view(s) for s in STATE["db"].list()]}


@app.post("/api/schedules")
async def add_schedule(body: SchedIn):
    if body.node not in CFG.node_names:
        raise HTTPException(400, "unknown node: %s" % body.node)
    s = STATE["db"].add(body.model, body.node, body.enabled, body.days,
                        body.on_time, body.off_time, body.note)
    return _sched_view(s)


@app.put("/api/schedules/{sid}")
async def update_schedule(sid: int, body: dict):
    allowed = {"enabled", "days", "on_time", "off_time", "note", "model", "node"}
    fields = {k: v for k, v in body.items() if k in allowed}
    s = STATE["db"].update(sid, **fields)
    if not s:
        raise HTTPException(404, "no such schedule")
    return _sched_view(s)


@app.delete("/api/schedules/{sid}")
async def delete_schedule(sid: int):
    if not STATE["db"].get(sid):
        raise HTTPException(404, "no such schedule")
    STATE["db"].delete(sid)
    return {"ok": True}


# ---------------- actions (load/unload) ----------------
async def _node_col(name):
    cols = STATE["collectors"]
    if name not in cols.nodes:
        raise HTTPException(404, "unknown node")
    return cols.nodes[name]


@app.post("/api/nodes/{name}/models/{model}/load")
async def load(name: str, model: str):
    col = await _node_col(name)
    return await col.load_model(model)


@app.post("/api/nodes/{name}/models/{model}/unload")
async def unload(name: str, model: str):
    col = await _node_col(name)
    return await col.unload_model(model)


@app.post("/api/nodes/{name}/unload-all")
async def unload_all(name: str):
    col = await _node_col(name)
    return await col.unload_all()


@app.get("/api/actions-log")
async def actions_log():
    return {"ts": _now_iso(), "log": STATE["sched"].log[-40:][::-1]}


# ---------------- static UI ----------------
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


_INDEX_HEADERS = {"Cache-Control": "no-store"}


@app.get("/")
async def index():
    return FileResponse(str(STATIC / "index.html"), headers=_INDEX_HEADERS)


@app.get("/index.html")
async def index2():
    return FileResponse(str(STATIC / "index.html"), headers=_INDEX_HEADERS)
