#!/usr/bin/env python3
"""
Logging
"""
import logging
import sys
import warnings

warnings.filterwarnings("ignore")


def get_logger(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configures the Root Logger so all modules inherit settings.
    """
    # 1. Get the ROOT logger (no name argument)
    log = logging.getLogger()

    # 2. Prevent duplicate logs if already configured
    if log.handlers:
        return log

    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    # 3. Console Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    # 4. File Handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    log.addHandler(sh)
    log.addHandler(fh)

    return log
