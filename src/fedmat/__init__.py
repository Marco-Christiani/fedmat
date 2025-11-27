"""Fedmat package initialization and exception handling utilities."""

from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING

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
