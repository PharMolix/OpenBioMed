from functools import wraps
from typing import Any, List, Tuple, Callable

from abc import ABC, abstractmethod

def serial_exec(func: Callable):
    @wraps(func)
    def wrapper(self, **kwargs):
        for key, value in kwargs.items():
            break
        if isinstance(value, list):
            outputs, msgs = [], []
            for i in range(len(value)):
                cur_kwargs = {}
                for key, value in kwargs.items():
                    cur_kwargs[key] = value[i]
                output, msg = func(self, **cur_kwargs)
                outputs.append(output)
                msgs.append(msg)
            return outputs, msgs
        else:
            output, msg = func(self, **kwargs)
            return [output], [msg]
    return wrapper

class Tool(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def print_usage(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs) -> Tuple[List[Any], List[Any]]:
        # The first argument should be directly passed as inputs for downstream tools
        # The second argument are used for communicating with front end
        raise NotImplementedError
