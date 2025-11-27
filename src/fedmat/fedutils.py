"""Utilities for federated learning model aggregation and reduction."""

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
    """Apply a reduction function to model parameters across multiple models.

    Parameters
    ----------
    inputs : list[nn.Module]
        List of input models to reduce
    reduce_fn : ReduceFn
        Function to reduce tensors
    output : nn.Module | None, optional
        Output model to accumulate results into. If None, uses deepcopy of first input, by default None

    Returns
    -------
    nn.Module
        Model with reduced parameters

    Raises
    ------
    ValueError
        If inputs is empty and output is None
    """
    if output is None:
        if len(inputs) == 0:
            raise ValueError("reduce needs at least one input model to determine output model shape")
        output = deepcopy(inputs[0])
    for name, parameter in output.named_parameters():
        parameter.copy_(reduce_fn([input.get_parameter(name) for input in inputs]))
    return output


def torch2reduce(fn: Callable[[Tensor], Tensor]) -> ReduceFn:
    """Convert a torch operation on a single tensor to a reduction function.

    Parameters
    ----------
    fn : Callable[[Tensor], Tensor]
        Function that operates on a single tensor

    Returns
    -------
    ReduceFn
        Function that stacks and applies fn along axis 0
    """
    return lambda tensors: fn(torch.stack(tensors), axis=0)
