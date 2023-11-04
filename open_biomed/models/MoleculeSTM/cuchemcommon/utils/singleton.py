# singleton.py

import logging

"""
Metaclass for singletons.
"""

logger = logging.getLogger(__name__)


class Singleton(type):
    """
    Ensures single instance of a class.

    Example Usage:
        class MySingleton(metaclass=Singleton)
            pass
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]
