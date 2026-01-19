"""Entry point for illama-manager server."""

import uvicorn

from .config import settings


def main() -> None:
    """Run the illama-manager server."""
    uvicorn.run(
        "illama_manager.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
