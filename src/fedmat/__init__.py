"""Fedmat package initialization and exception handling utilities."""

from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop
    from types import TracebackType

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[stdout_handler, stderr_handler],
)


def _log_main_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    """Log uncaught exceptions from the main thread."""
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    """Log uncaught exceptions from worker threads."""
    logging.error(
        "Uncaught exception in thread %s",
        args.thread.name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _log_async_exception(
    _: AbstractEventLoop,
    context: dict[str, BaseException | str],
) -> None:
    """Log uncaught exceptions from asyncio tasks."""
    msg = context.get("message", "asyncio task exception")
    exc = context.get("exception")
    logging.error(msg, exc_info=exc)


def install_exception_handlers(threading_: bool = False, asyncio_: bool = False) -> None:
    """Install custom exception handlers for main, threading, and asyncio exceptions.

    Parameters
    ----------
    threading_ : bool, optional
        If True, install handler for uncaught thread exceptions, by default False
    asyncio_ : bool, optional
        If True, install handler for asyncio task exceptions, by default False
    """
    sys.excepthook = _log_main_exception
    if threading_:
        threading.excepthook = _log_thread_exception
    if asyncio_:
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.get_exception_handler() is None:
            loop.set_exception_handler(_log_async_exception)


_log_ctx = threading.local()


def set_log_context(**kwargs: Any) -> None:
    """Attach arbitrary metadata to log records (thread-local)."""
    for k, v in kwargs.items():
        setattr(_log_ctx, k, v)


def clear_log_context() -> None:
    """Clear all injected log context."""
    _log_ctx.__dict__.clear()


def install_global_log_context() -> None:
    """Inject context into all log handlers and all future handlers."""
    root = logging.getLogger()

    filt = _ContextFilter()
    for h in root.handlers:
        h.addFilter(filt)

    # Patch future handlers
    _patch_logger_add_handler(filt)

    # Patch existing formatters so they accept contextual fields
    for h in root.handlers:
        if h.formatter is not None:
            _patch_formatter(h.formatter)


class _ContextFilter(logging.Filter):
    """Filter that injects thread-local context into all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        # defaults if nothing has been set yet
        # if "rank" not in _log_ctx.__dict__:
        #     setattr(record, "rank", "main")
        # if "world_size" not in _log_ctx.__dict__:
        #     setattr(record, "world_size", 1)

        # # inject all context keys
        # for k, v in _log_ctx.__dict__.items():
        #     setattr(record, k, v)

        # return True
        defaults = {
            "rank": "-",
            "world_size": "-",
            "backend": "-",
        }

        # defaults if field is missing
        for k, v in defaults.items():
            if not hasattr(record, k):
                setattr(record, k, v)

        # inject thread-local overrides
        for k, v in _log_ctx.__dict__.items():
            setattr(record, k, v)

        return True


def _patch_logger_add_handler(filt: logging.Filter) -> None:
    """Monkey-patch Logger.addHandler so future handlers inherit the filter."""
    orig_add_handler = logging.Logger.addHandler

    def add_handler_with_patch(self, hdlr):
        hdlr.addFilter(filt)
        if hdlr.formatter is not None:
            _patch_formatter(hdlr.formatter)
        orig_add_handler(self, hdlr)

    logging.Logger.addHandler = add_handler_with_patch


def _patch_formatter(fmt: logging.Formatter) -> None:
    """Ensure formatter templates include %(rank)s, %(world_size)s, etc."""
    prefix = "[rank=%(rank)s world=%(world_size)s backend=%(backend)s] "

    for attr in ("_fmt", "_style"):
        obj = getattr(fmt, attr, None)
        if not obj:
            continue

        # Case: standard formatter._fmt is a string.
        if isinstance(obj, str):
            if "%(rank)" not in obj:
                setattr(fmt, attr, prefix + obj)
            continue

        # Case: old-style or new-style _style._fmt is a string.
        style_fmt = getattr(obj, "_fmt", None)
        if isinstance(style_fmt, str) and "%(rank)" not in style_fmt:
            obj._fmt = prefix + style_fmt
