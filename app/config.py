"""Configuration loading: config.env (KEY=VALUE) + nodes.json + .dashro.pass."""
import os
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent.parent  # project root


def _load_envfile(path: pathlib.Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _load_nodes(path: pathlib.Path):
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _norm_model(s: str) -> str:
    """Normalize a model name for comparison: lowercase, alnum only."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def save_nodes(nodes):
    """Persist the live node list back to nodes.json (atomic)."""
    p = BASE / "nodes.json"
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(nodes, f, indent=2)
        f.write("\n")
    os.replace(tmp, p)


class Config:
    def __init__(self):
        _load_envfile(BASE / "config.env")
        self.host = os.environ.get("DASH_HOST", "0.0.0.0")
        self.port = int(os.environ.get("DASH_PORT", "8090"))
        self.poll_interval = float(os.environ.get("POLL_INTERVAL", "1.5"))
        self.node_timeout = float(os.environ.get("NODE_TIMEOUT", "6"))

        self.pg_host = os.environ.get("PG_HOST", "127.0.0.1")
        self.pg_port = int(os.environ.get("PG_PORT", "15432"))
        self.pg_user = os.environ.get("PG_USER", "dashro")
        self.pg_db = os.environ.get("PG_DB", "litellm")
        self.pg_password = os.environ.get("PG_PASSWORD", "")
        if not self.pg_password:
            pf = BASE / ".dashro.pass"
            if pf.exists():
                self.pg_password = pf.read_text().strip()

        self.local_prefixes = [
            p.strip()
            for p in os.environ.get("LOCAL_PREFIXES", "http://192.168.").split(",")
            if p.strip()
        ]
        # Model names that are always counted as LOCAL, regardless of api_base
        # (spend-log api_base is often empty, which would otherwise classify
        # them as online). Compared normalized: lowercase, alnum only.
        self.local_models = [
            _norm_model(p)
            for p in os.environ.get("LOCAL_MODELS", "").split(",")
            if p.strip()
        ]

        self.data_dir = BASE / os.environ.get("DATA_DIR", "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.schedule_enforce = os.environ.get("SCHEDULE_ENFORCE", "1") == "1"
        self.schedule_interval = float(os.environ.get("SCHEDULE_INTERVAL", "20"))
        self.offline_grace = float(os.environ.get("OFFLINE_GRACE", "60"))

        self.nodes = _load_nodes(BASE / "nodes.json")
        for n in self.nodes:
            n["base"] = "http://%s:%s" % (n["ip"], n.get("port", 8080))
        self.node_names = [n["name"] for n in self.nodes]

    def pg_dsn(self) -> str:
        return "postgresql://%s:%s@%s:%s/%s" % (
            self.pg_user,
            self.pg_password,
            self.pg_host,
            self.pg_port,
            self.pg_db,
        )


CFG = Config()
