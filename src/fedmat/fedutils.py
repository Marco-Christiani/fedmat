from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import torch
from torch import Tensor, nn

ReduceFn = Callable[[list[Tensor]], Tensor]


def reduce(
    inputs: list[nn.Module],
    reduce_fn: ReduceFn,
    output: nn.Module | None = None,
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
