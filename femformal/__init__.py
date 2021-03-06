import logging

logger = logging.getLogger("femformal")
logger.addHandler(logging.NullHandler())

stl_logger = logging.getLogger("stlmilp")
for handler in stl_logger.handlers:
    stl_logger.removeHandler(handler)
stl_logger.addHandler(logging.NullHandler())
stl_logger = logging.getLogger("stl_milp_encode")
for handler in stl_logger.handlers:
    stl_logger.removeHandler(handler)
stl_logger.addHandler(logging.NullHandler())

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

import matplotlib.style
import matplotlib as mpl

mpl.style.use("classic")

import sys
import os

FOCUSED = ":" in sys.argv[-1]

if "nose" in sys.modules.keys() and FOCUSED:
    import logging.config

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "debug_formatter": {
                    "format": "%(levelname).1s %(module)s:%(lineno)d:%(funcName)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "debug_formatter",
                }
            },
            "loggers": {
                "femformal": {
                    "handlers": ["console"],
                    "level": "DEBUG",
                    "propagate": True,
                }
            },
        }
    )
