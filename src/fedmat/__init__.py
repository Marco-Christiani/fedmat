import logging
import sys
import threading

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


def _log_main_exception(exc_type, exc_value, exc_traceback):
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def _log_thread_exception(args):
    logging.error(
        "Uncaught exception in thread %s",
        args.thread.name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _log_async_exception(_, context):
    msg = context.get("message", "asyncio task exception")
    exc = context.get("exception")
    logging.error(msg, exc_info=exc)


def install_exception_handlers(threading_: bool = False, asyncio_: bool = False):
    sys.excepthook = _log_main_exception
    if threading_:
        threading.excepthook = _log_thread_exception
    if asyncio_:
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.get_exception_handler() is None:
            loop.set_exception_handler(_log_async_exception)
