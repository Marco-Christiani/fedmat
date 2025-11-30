from abc import ABC
from typing import List, Tuple

import torch
from torch import nn, Tensor

def vectorize(m: nn.Module, names: List[str]) -> Tensor:
    return torch.cat([m.get_parameter(name).flatten() for name in names])

def squared_distance_matrix(a: List[nn.Module], b: List[nn.Module]) -> Tensor:
    """Calculates the distance matrix of every module in a to every module in b."""

    if len(a) != len(b):
        raise ValueError("Bipartite matching needs a bipartite graph")
    
    if len(a) == 0:
        return torch.zeroes((0,0))

    names = [name for names, _ in a[0].named_parameters()]
    a_vec = torch.stack([vectorize(m, names) for m in a])
    b_vec = torch.stack([vectorize(m, names) for m in b])
    return torch.cdist(a_vec, b_vec)

class Matcher(ABC):
    def match(a: List[nn.Module], b: List[nn.Module]) -> List[Tuple[int, int]]:
        """Match the client blocks to each other to minimize some distance metric."""
        # TODO: gooder documentation
        pass

class GreedyMatcher(Matcher):
    def match(a: List[nn.Module], b: List[nn.Module]) -> List[Tuple[int, int]]:
        """Le Greedy"""
        pass

class HungarianMatcher(Matcher):
    def match(a: List[nn.Module], b: List[nn.Module]) -> List[Tuple[int, int]]:
        """Le Hungarian ala FedMA"""
        pass
