import logging
import sys
import os
from typing import Dict, List
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torch

mpl.use("Agg")

import threading
from tabulate import tabulate
from inspect import getframeinfo, stack

from loguru import logger
logger.remove()


class LoggerSubscriber:
    def __init__(self, **kwargs) -> None:
        pass

    def log_scalar(self, **kwargs):
        return

    def log_figure(self, **kwargs):
        return

    def log_torch_module(self, **kwargs):
        return

    def log_text(self, **kwargs):
        return

    def log_embedding(self, **kwargs):
        return

    def log_spec_text(self, **kwargs):
        return

    def log_table(self, **kwargs):
        return

    def log_video(self, **kwargs):
        return

    def log_html(self, **kwargs):
        return



class LoggerObserver(object):
    """Logger Oberserver Degisn Pattern
    notifies every subscribers when .log() is called
    """

    SCALAR = "scalar"
    FIGURE = "figure"
    TORCH_MODULE = "torch_module"
    TEXT = "text"
    SPECIAL_TEXT = "special_text"
    EMBED = "embedding"
    TABLE = "table"
    VIDEO = "video"
    HTML = "html"

    WARN = logging.WARN
    ERROR = logging.ERROR
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    CRITICAL = logging.CRITICAL
    SUCCESS = "SUCCESS"

    instances = {}
    _lock = threading.Lock()

    def __new__(cls, name=None, *args, **kwargs):
        with cls._lock:
            if name is None:
                name = str(os.getpid())
            if name in LoggerObserver.instances.keys():
                return LoggerObserver.instances[name]

            return object.__new__(cls, *args, **kwargs)

    def __init__(self, name) -> None:
        if getattr(self, "_initialized", False):
            return
        self.subscriber = []
        self.name = name

        # Init with a stdout logger

        logger = StdoutLogger(name=self.name, debug=True)
        self.subscribe(logger)

        LoggerObserver.instances[name] = self
        self._initialized = True

    def __del__(self):
        self.subscriber.clear()
        if self.name in LoggerObserver.instances.keys():
            LoggerObserver.instances.pop(self.name)

    @classmethod
    def getLogger(cls, name):
        if name in LoggerObserver.instances.keys():
            return LoggerObserver.instances[name]

        return cls(name)

    def subscribe(self, subscriber: LoggerSubscriber):
        self.subscriber.append(subscriber)

    def log(self, logs: List[Dict]):
        
        # Support distributed logging
        is_master = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
        if not is_master:
            return
        
        for subscriber in self.subscriber:
            for log in logs:
                tag = log["tag"]
                value = log["value"]
                log_type = log["type"] if "type" in log.keys() else get_type(value)
                kwargs = log["kwargs"] if "kwargs" in log.keys() else {}

                if log_type == LoggerObserver.SCALAR:
                    subscriber.log_scalar(tag=tag, value=value, **kwargs)

                if log_type == LoggerObserver.FIGURE:
                    subscriber.log_figure(tag=tag, value=value, **kwargs)

                if log_type == LoggerObserver.TORCH_MODULE:
                    subscriber.log_torch_module(tag=tag, value=value, **kwargs)

                if log_type == LoggerObserver.TEXT:
                    subscriber.log_text(tag=tag, value=value, **kwargs)

                if log_type == LoggerObserver.EMBED:
                    subscriber.log_embedding(tag=tag, value=value, **kwargs)

                if log_type == LoggerObserver.SPECIAL_TEXT:
                    subscriber.log_spec_text(tag=tag, value=value, **kwargs)

                if log_type == LoggerObserver.TABLE:
                    subscriber.log_table(tag=tag, value=value, **kwargs)

                if log_type == LoggerObserver.VIDEO:
                    subscriber.log_video(tag=tag, value=value, **kwargs)

                if log_type == LoggerObserver.HTML:
                    subscriber.log_html(tag=tag, value=value, **kwargs)

    def text(self, *value, level=logging.INFO):
        """
        Text logging
        """
        caller = getframeinfo(stack()[1][0])
        function_name = stack()[1][3]
        filename = "//".join(caller.filename.split("ehrpfn")[1:])[
            1:
        ]  # split filename based on project name
        lineno = caller.lineno

        texts = []
        for v in value:
            if isinstance(v, dict):
                texts.append(json.dumps(v, indent=4, default=str))
            else:
                texts.append(str(v))
        value = ' '.join(texts)

        self.log(
            [
                {
                    "tag": "stdout",
                    "value": value,
                    "type": LoggerObserver.TEXT,
                    "kwargs": {
                        "level": level,
                        "lineno": lineno,
                        "filename": filename,
                        "funcname": function_name,
                    },
                }
            ]
        )

    def __repr__(self) -> str:
        table_headers = ["Subscribers"]
        table = tabulate(
            [[type(i).__name__] for i in self.subscriber],
            headers=table_headers,
            tablefmt="fancy_grid",
        )
        return "Logger subscribers: \n" + table



class BaseTextLogger(LoggerSubscriber):
    """
    Logger class for showing text in prompt and file
    For more documents, look into https://docs.python.org/3/library/logging.html

    Usage:
        from modules.logger import BaseTextLogger
        LOGGER = BaseTextLogger.init_logger(__name__)

    """

    message_format = """<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <light-black>{extra[filename]}</light-black>:<light-black>{extra[funcname]}</light-black>:<light-black>{extra[lineno]}</light-black> - <level>{message}</level>"""

    def __init__(self, name):
        self.name = name
        self._instance_id = str(id(self))

    def log_text(self, tag, value, level=LoggerObserver.DEBUG, **kwargs):
        if level == LoggerObserver.WARN:
            logger.warning(value)

        if level == LoggerObserver.INFO:
            logger.info(value)

        if level == LoggerObserver.ERROR:
            logger.error(value)

        if level == LoggerObserver.DEBUG:
            logger.debug(value)

        if level == LoggerObserver.SUCCESS:
            logger.success(value)


class FileLogger(BaseTextLogger):
    """
    Logger class for showing text in prompt and file
    For more documents, look into https://docs.python.org/3/library/logging.html

    Usage:
        from modules.logger import FileLogger
        LOGGER = FileLogger.init_logger(__name__)

    """

    def __init__(self, name, logdir, rotation="10 MB", debug=False):
        self.logdir = logdir
        self.filename = f"{self.logdir}/log.txt"
        super().__init__(name)

        if debug:
            level = "DEBUG"
        else:
            level = "INFO"

        self._handler_id = logger.add(
            self.filename,
            rotation=rotation,
            level=level,
            filter=lambda record: record["extra"].get("handler_id") == self._instance_id,
        )

    def log_text(self, tag, value, level=LoggerObserver.DEBUG, **kwargs):
        filename = kwargs.get("filename", None)
        funcname = kwargs.get("funcname", None)
        lineno = kwargs.get("lineno", None)
        with logger.contextualize(
            handler_id=self._instance_id, filename=filename, funcname=funcname, lineno=lineno
        ):
            return super().log_text(tag, value, level, **kwargs)

    def __del__(self):
        if hasattr(self, "_handler_id"):
            try:
                logger.remove(self._handler_id)
            except ValueError:
                pass


class StdoutLogger(BaseTextLogger):
    """
    Logger class for showing text in prompt and file
    For more documents, look into https://docs.python.org/3/library/logging.html

    Usage:
        from modules.logger import StdoutLogger
        LOGGER = StdoutLogger.init_logger(__name__)

    """

    def __init__(self, name, debug=False):
        super().__init__(name)

        if debug:
            level = "DEBUG"
        else:
            level = "INFO"

        self._handler_id = logger.add(
            sys.stdout,
            backtrace=True,
            diagnose=True,
            level=level,
            format=self.message_format,
            filter=lambda record: record["extra"].get("handler_id") == self._instance_id,
        )

    def log_text(self, tag, value, level=LoggerObserver.DEBUG, **kwargs):
        filename = kwargs.get("filename", None)
        funcname = kwargs.get("funcname", None)
        lineno = kwargs.get("lineno", None)
        with logger.contextualize(
            handler_id=self._instance_id, filename=filename, funcname=funcname, lineno=lineno
        ):
            return super().log_text(tag, value, level, **kwargs)

    def __del__(self):
        if hasattr(self, "_handler_id"):
            try:
                logger.remove(self._handler_id)
            except ValueError:
                pass


def get_type(value):
    if isinstance(value, torch.nn.Module):
        return LoggerObserver.TORCH_MODULE
    if isinstance(value, mpl.figure.Figure) or isinstance(value, go.Figure):
        return LoggerObserver.FIGURE
    if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
        if len(value.shape) == 2:
            return LoggerObserver.EMBED
    if isinstance(value, (int, float)):
        return LoggerObserver.SCALAR
    if isinstance(value, str):
        if value.endswith(".html"):
            return LoggerObserver.HTML
        else:
            return LoggerObserver.TEXT
    else:
        raise ValueError(f"Fail to log undefined type: {type(value)}")

