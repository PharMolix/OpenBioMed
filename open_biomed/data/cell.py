from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

import copy
from datetime import datetime
import numpy as np
import os
import pickle
try:
    import scanpy as sc
except ImportError:
    raise ImportError("Install scanpy to use cell APIs: pip install scanpy")
import re

from open_biomed.tools.base_tool import Tool
from open_biomed.data.text import Text

class Cell:
    def __init__(self) -> None:
        super().__init__()
        self.anndata = None
        self.sequence = None

    @classmethod
    def from_anndata(cls, anndata: sc.AnnData) -> Self:
        cell = cls()
        cell.anndata = anndata
        return cell

    @classmethod
    def from_sequence(cls, sequence: str) -> Self:
        cell = cls()
        cell.sequence = sequence
        return cell
    
    def __str__(self) -> str:
        return 'cell'