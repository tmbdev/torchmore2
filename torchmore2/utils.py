import sys
import warnings
from typing import Callable, List, Tuple, Type

import numpy as np
import torch
from torch import Tensor, autograd, nn
from torch.nn import functional as F


def model_device(model):
    """Find the device of a model."""
    return next(model.parameters()).device


def DEPRECATED(f: Callable):
    def g(*args, **kw):
        raise Exception("DEPRECATED")
        return f(*args, **kw)

    return g


def deprecated(f: Callable):
    def g(*args, **kw):
        warnings.warn("deprecated", DeprecationWarning)
        return f(*args, **kw)

    return g
