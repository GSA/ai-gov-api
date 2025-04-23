import logging
import sys
from typing import Optional

import structlog
from structlog.contextvars import  merge_contextvars
from app.config.settings import settings


def setup_structlog():
    """Configure structlog for FastAPI application logging"""
    log_level = settings.log_level.upper()
    
    base = [
        merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    # Pretty colorful console output in development
    if settings.env == "dev":
        processors = base
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # JSON formatting for production
        processors = base + [structlog.processors.format_exc_info]
        renderer = structlog.processors.JSONRenderer()

    
    structlog.configure(
        processors=processors + [renderer],
        context_class=dict,
        # this causes stuctlog to be just a thin wrapper around logging
        # so fastAPI and other logs use out prefered format
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level)),
        logger_factory=structlog.stdlib.LoggerFactory(), 
        cache_logger_on_first_use=True,
    )
    
    # https://www.structlog.org/en/stable/standard-library.html
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level)
    )
    

def get_logger(name: Optional[str] = None):
    """This just adds an abstraction layer to make it easier to swap later if needed"""
    return structlog.get_logger(name)

