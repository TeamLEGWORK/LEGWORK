"""Logging setup for LEGWORK

Every message that LEGWORK shows the user is sent through a logger called ``LEGWORK``, which is created and
attached to a handler when the package is imported. Messages are prefixed with a bold ``LEGWORK <level>``,
which is coloured yellow for warnings and red for errors when the output is going to a terminal.

The messages can be silenced (or made more verbose) by changing the level of the logger, e.g. to hide
anything that isn't an error::

    import logging
    legwork.logger.setLevel(logging.ERROR)
"""

import logging
import sys

__all__ = ["logger"]

# ANSI escape codes used to style the prefix of each message when writing to a terminal
_BOLD = "\033[1m"
_RESET = "\033[0m"
_LEVEL_COLOURS = {
    logging.WARNING: "\033[33m",        # yellow
    logging.ERROR: "\033[31m",          # red
    logging.CRITICAL: "\033[31m",       # red
}


class LegworkFormatter(logging.Formatter):
    """Formatter that prefixes each message with a bold (and coloured) ``LEGWORK <level>``

    Parameters
    ----------
    stream : `file-like`, optional
        The stream that the messages are written to. This is only used to check whether the output
        supports colours, no colours are used if it isn't a terminal (or isn't supplied).
    """

    def __init__(self, stream=None):
        super().__init__(fmt="%(prefix)s: %(message)s")
        self._stream = stream

    def use_colour(self):
        """Whether the messages should be coloured (only if we are writing to a terminal)"""
        return self._stream is not None and getattr(self._stream, "isatty", lambda: False)()

    def format(self, record):
        prefix = "LEGWORK {}".format(record.levelname.lower())
        if self.use_colour():
            prefix = "{}{}{}{}".format(_BOLD, _LEVEL_COLOURS.get(record.levelno, ""), prefix, _RESET)
        record.prefix = prefix
        return super().format(record)


logger = logging.getLogger("LEGWORK")

# attach a handler so that users see the messages without needing to configure logging themselves
if not logger.handlers:
    # messages go to stdout since that's where they went when they were printed
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(LegworkFormatter(stream=sys.stdout))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # don't pass the messages up to the root logger as well, they'd be shown twice
    logger.propagate = False
