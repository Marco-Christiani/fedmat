from abc import ABC
from typing import List, Tuple

import torch
from torch import nn, Tensor

def module_names(m: nn.Module) -> List[str]:
    return [name for name, _ in m.named_parameters()]

def vectorize(m: nn.Module, names: List[str]) -> Tensor:
    """Vectorize a module according to the order given in names."""
    return torch.cat([m.get_parameter(name).flatten() for name in names])

class Matcher(ABC):
    def match(clients: Tensor) -> Tensor:
        """
        clients: C x m x d array of vectorized attention heads, from the C clients each with m attention heads
        returns: C x m matrix of cluster ids ranging from [0, m), with the first row being in order [0, 1, ..., m-1]
        """
        # TODO: gooder documentation
        pass

class GreedyMatcher(Matcher):
    def match(clients: Tensor) -> Tensor:
        """Le Greedy"""
        pass

class HungarianMatcher(Matcher):
    def match(clients: Tensor) -> Tensor:
        """Le Hungarian ala FedMA"""
        pass
