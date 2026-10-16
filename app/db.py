"""SQLite store for model schedules (persistent across restarts)."""
import sqlite3
import threading
import time

# days bitmask: index 0=Mon .. 6=Sun, char '1'/'0'. "1111111" = every day.
ALL_DAYS = "1111111"


class DB:
    def __init__(self, path):
        self.path = str(path)
        self._lock = threading.RLock()
        self._init()

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def _init(self):
        with self._lock, self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS schedules(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    node  TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    days TEXT NOT NULL DEFAULT '1111111',
                    on_time TEXT NOT NULL DEFAULT '00:00',
                    off_time TEXT NOT NULL DEFAULT '23:59',
                    note TEXT NOT NULL DEFAULT '',
                    created_at REAL,
                    updated_at REAL,
                    last_action TEXT NOT NULL DEFAULT '',
                    last_state INTEGER NOT NULL DEFAULT 0
                )"""
            )
            c.commit()

    @staticmethod
    def _row(r):
        return {
            "id": r["id"],
            "model": r["model"],
            "node": r["node"],
            "enabled": bool(r["enabled"]),
            "days": r["days"],
            "on_time": r["on_time"],
            "off_time": r["off_time"],
            "note": r["note"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "last_action": r["last_action"],
            "last_state": r["last_state"],
        }

    def list(self):
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT * FROM schedules ORDER BY node, model, on_time"
            ).fetchall()
        return [self._row(r) for r in rows]

    def get(self, sid):
        with self._lock, self._conn() as c:
            r = c.execute("SELECT * FROM schedules WHERE id=?", (sid,)).fetchone()
        return self._row(r) if r else None

    def add(self, model, node, enabled=True, days=ALL_DAYS, on_time="00:00",
            off_time="23:59", note=""):
        now = time.time()
        with self._lock, self._conn() as c:
            cur = c.execute(
                """INSERT INTO schedules(model,node,enabled,days,on_time,off_time,
                    note,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (model, node, 1 if enabled else 0, days, on_time, off_time, note,
                 now, now),
            )
            c.commit()
            sid = cur.lastrowid
        return self.get(sid)

    def update(self, sid, **fields):
        allowed = {"enabled", "days", "on_time", "off_time", "note", "model", "node"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append("%s=?" % k)
                vals.append(v)
        if not sets:
            return self.get(sid)
        sets.append("updated_at=?")
        vals.append(time.time())
        vals.append(sid)
        with self._lock, self._conn() as c:
            c.execute("UPDATE schedules SET %s WHERE id=?" % ",".join(sets), vals)
            c.commit()
        return self.get(sid)

    def set_state(self, sid, last_state, last_action=""):
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE schedules SET last_state=?, last_action=? WHERE id=?",
                (last_state, last_action, sid),
            )
            c.commit()

    def delete(self, sid):
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM schedules WHERE id=?", (sid,))
            c.commit()
