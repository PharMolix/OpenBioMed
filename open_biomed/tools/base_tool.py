from typing import Any, List, Tuple, Callable

from abc import ABC, abstractmethod

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
