from copy import deepcopy

import torch
from torch import Tensor
from torch import nn
from typing import List, Optional
from collections.abc import Callable

ReduceFn = Callable[[List[Tensor]], Tensor]

def reduce(
        inputs: List[nn.Module],
        reduce_fn: ReduceFn,
        output: Optional[nn.Module] = None,
        ) -> nn.Module:
    if output is None:
        if len(inputs) == 0:
            raise ValueError("reduce needs at least one input model to determine output model shape")
        output = deepcopy(inputs[0])
    for name, parameter in output.named_parameters():
        parameter.copy_(reduce_fn([input.get_parameter(name) for input in inputs]))
    return output

def torch2reduce(fn: Callable[[Tensor], Tensor]) -> ReduceFn:
    return lambda tensors: fn(torch.stack(tensors), axis=0)

def replicate(
        input: nn.Module,
        outputs: int | List[nn.Module]
        ) -> List[nn.Module]:
    if isinstance(outputs, int):
        return [deepcopy(input) for _ in range(outputs)] 
    for name, parameter in input.named_parameters():
        for output in outputs:
            output.get_parameter(name).copy_(parameter)
