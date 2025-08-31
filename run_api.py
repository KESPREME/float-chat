"""Run the FloatChat API server."""

import uvicorn
from app.api.main import app
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
