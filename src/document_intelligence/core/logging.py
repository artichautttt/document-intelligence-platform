"""Configuration du logging structuré (JSON) pour toute l'application."""

import logging
import sys

import structlog

from document_intelligence.core.config import settings


def configure_logging() -> None:
    """Configure structlog pour émettre des logs JSON structurés sur stdout.

    Doit être appelé une seule fois, au démarrage du processus (script, worker, API).
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Retourne un logger structuré nommé, à utiliser dans chaque module."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
