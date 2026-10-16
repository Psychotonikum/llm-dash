#!/usr/bin/env python3
"""llamaDash entrypoint: uvicorn server + background collector/scheduler."""
import uvicorn
from app.config import CFG

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=CFG.host,
        port=CFG.port,
        log_level="info",
        lifespan="on",
    )
