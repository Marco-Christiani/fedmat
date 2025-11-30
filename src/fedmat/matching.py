from abc import ABC
from torch import nn
from typing import List, Tuple

class Matcher(ABC):
    def match(client_blocks: List[nn.Module]) -> List[Tuple[int, int]]:
        """Match the client blocks to each other to minimize some distance metric."""
        # TODO: gooder documentation
        pass

class GreedyMatcher(Matcher):
    def match(client_blocks: List[nn.Module]) -> List[Tuple[int, int]]:
        """Le Greedy"""
        pass

class PseudoHungarianMatcher(Matcher):
    def match(client_blocks: List[nn.Module]) -> List[Tuple[int, int]]:
        """Le Hungarian ala FedMA"""
        pass

class PerfectMatcher(Matcher):
    def match(client_blocks: List[nn.Module]) -> List[Tuple[int, int]]:
        """Use some quantum error correction package to do matching"""
        pass
