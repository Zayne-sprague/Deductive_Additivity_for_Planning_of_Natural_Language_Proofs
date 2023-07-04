from typing import Dict
from pathlib import Path
import json
import os
import logging


class LoggerModule:
    """
    Helper class for handling log messages
    """

    console: bool
    log_dir: Path

    logs: Dict[str, logging.Logger]

    def __init__(
            self,
            console: bool = True,
            log_dir: Path = None,
            remove_existing_logs: bool = True
    ):
        self.logs = {}
        self.log_dir = log_dir
        self.console = console

        self.build_logger('base', self.log_dir, remove_existing_logs)

    def build_logger(self, name: str, log_dir: Path, remove_existing: bool = True) -> logging.Logger:
        """Allows you to create a log with a name and directory -- useful for splitting across cuda devices or cores."""

        log = logging.getLogger(name)
        log.setLevel(logging.INFO)

        if self.console:
            c_handler = logging.StreamHandler()
            c_handler.setLevel(logging.INFO)
            c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(c_format)
            log.addHandler(c_handler)

        if log_dir:
            log_dir.mkdir(exist_ok=True, parents=True)

            info_file = log_dir / 'info.log'
            error_file = log_dir / 'error.log'

            if remove_existing:
                if info_file.is_file():
                    os.remove(str(info_file))
                if error_file.is_file():
                    os.remove(str(error_file))

            f_handler = logging.FileHandler(str(info_file))
            f_handler.setLevel(logging.DEBUG)
            f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            f_handler.setFormatter(f_format)
            log.addHandler(f_handler)

            f_handler = logging.FileHandler(str(error_file))
            f_handler.setLevel(logging.ERROR)
            f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            f_handler.setFormatter(f_format)
            log.addHandler(f_handler)

        self.logs[name] = log
        return self.logs[name]

    @property
    def logger(self) -> logging.Logger:
        return self.logs['base']
