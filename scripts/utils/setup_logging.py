import logging
import warnings

import structlog

logging.basicConfig(level=logging.CRITICAL)


def setup_logging():  # pragma no cover
    """Configures logging."""

    def _rename_event_key(_, __, event_dict):
        """Renames events to what our logging system expects."""
        event_dict["message"] = event_dict.pop("event")
        return event_dict

    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.contextvars.merge_contextvars,
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _rename_event_key,
    ]
    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()

    # Removing existing handlers so they don't emit logs.
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.CRITICAL)

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.CRITICAL)

    # warnings is from xgboost
    warnings.filterwarnings("ignore")
