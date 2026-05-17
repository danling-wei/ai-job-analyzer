"""Logging configuration based on loguru."""

from __future__ import annotations

import logging
import sys
from types import FrameType

from loguru import logger

from .config import get_settings

_CONFIGURED = False


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin shim
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> None:
    """Initialise application-wide logging exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=settings.environment == "dev",
    )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).handlers = [_InterceptHandler()]
        logging.getLogger(noisy).propagate = False

    _CONFIGURED = True
