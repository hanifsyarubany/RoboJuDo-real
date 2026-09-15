import logging
import logging.handlers
import queue
from pathlib import Path

import colorlog
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that works well with tqdm progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# Bound generous enough to absorb even a multi-minute stalled consumer at this project's typical
# log rate (a handful of lines/sec) without dropping anything in practice -- it exists purely as a
# memory cap, not a normal-operation limit. See _DropOldestQueueHandler's docstring for why dropping
# beats blocking here.
_LOG_QUEUE_MAXSIZE = 5000


class _DropOldestQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler whose enqueue() never blocks: drops the OLDEST queued record on overflow instead.

    Same rationale/pattern as controller/utils/joystick.py's _put_event_dropping_oldest: a bounded
    queue.put() with no guard can block its CALLER for as long as the consumer is stalled. Diagnosed
    via a faulthandler stack dump from a real robot freeze (82s, then 45s in an earlier incident) --
    the MAIN CONTROL-LOOP THREAD was stuck inside logger.info() -> TqdmLoggingHandler.emit() ->
    tqdm.write() -> a raw stdout write(), invoked synchronously from a routine rate-diagnostic log
    call. A laggy SSH terminal, a scroll-locked/paused pane, or a stuck disk on the FileHandler side
    can each stall that write for an unbounded time, and until this queue existed, that meant the
    control loop itself -- not just logging -- froze for exactly as long.
    """

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(record)


def setup_logger(name: str = "robojudo") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # to avoid duplicate logs from phc

    color_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    # console_handler = logging.StreamHandler(sys.stdout)
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(color_formatter)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s", "%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Both real handlers above do BLOCKING I/O with no timeout (tqdm.write() -> stdout write();
    # FileHandler -> a disk write()). Routing every logger.<level>() call from EVERY thread --
    # including the robot's own real-time control loop -- through a bounded, non-blocking queue
    # instead means that I/O can only ever stall this dedicated listener thread, never the caller.
    # See _DropOldestQueueHandler's docstring for the incident this fixes.
    log_queue: queue.Queue = queue.Queue(maxsize=_LOG_QUEUE_MAXSIZE)
    queue_handler = _DropOldestQueueHandler(log_queue)
    logger.addHandler(queue_handler)

    listener = logging.handlers.QueueListener(log_queue, console_handler, file_handler, respect_handler_level=True)
    listener.start()  # starts its own daemon thread -- never blocks process exit
    logger._robojudo_queue_listener = listener  # kept alive via the logger; no explicit stop() path
    # needed today since nothing currently shuts this logger down independently of the whole process.

    return logger
