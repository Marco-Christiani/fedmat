from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor, nn

from fedmat.train_utils import (
    ModelFlatMetadata,
    ModelReshaper,
    StateDict,
    build_flat_metadata,
    clone_state_dict,
    flatten_state_dict,
    unflatten_state_dict,
)

# Fixtures ---------------------------------------------------------------------


TinyNetState = tuple[nn.Module, StateDict, ModelFlatMetadata]


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(4, 3)
        self.bn = nn.BatchNorm1d(3)


@pytest.fixture
def tinynet_state() -> TinyNetState:
    """TinyNet + state dict + metadata.

    Note: state dict includes both float and int tensors (e.g. num_batches_tracked),
      for testing dtype behavior realistically.
    """
    torch.manual_seed(0)
    model = TinyNet()
    state_ = model.state_dict()
    assert isinstance(state_, OrderedDict)
    state: OrderedDict = state_  # yay, python! ridiculous.

    # Fill tensors with deterministic ascending values, preserving dtype.
    for name, t in state.items():
        numel = t.numel()
        if numel == 0:
            state[name] = torch.empty_like(t)
        else:
            if t.dtype.is_floating_point:
                vals = torch.arange(numel, dtype=torch.float32).view_as(t)
                state[name] = vals.to(dtype=t.dtype)
            else:
                vals = torch.arange(numel, dtype=torch.int64).view_as(t)
                state[name] = vals.to(dtype=t.dtype)

    meta = build_flat_metadata(state)
    return model, state, meta


#  Tests -----------------------------------------------------------------------


def test_metadata_matches(tinynet_state: TinyNetState) -> None:
    _, state, meta = tinynet_state

    assert list(state.keys()) == list(meta.names)
    assert [t.shape for t in state.values()] == list(meta.shapes)
    assert [t.dtype for t in state.values()] == list(meta.dtypes)
    assert [t.numel() for t in state.values()] == list(meta.numels)
    assert meta.total_numel == sum(t.numel() for t in state.values())


def test_roundtrip_flat_unflat(tinynet_state: TinyNetState) -> None:
    _, state, meta = tinynet_state

    flat = flatten_state_dict(state, meta)
    restored = unflatten_state_dict(flat, meta)

    for k in state:
        # compare in common dtype
        orig = state[k].to(dtype=torch.float32)
        new = restored[k].to(dtype=torch.float32)
        assert torch.allclose(orig, new)


def test_flatten_order_is_metadata_driven(tinynet_state: TinyNetState) -> None:
    """Make sure metadata control is correct.

    Even if the input OrderedDict has a different insertion order,
    flatten_state_dict should still follow metadata.names for *value order*.

    However, our implementation chooses the result dtype from the first
    tensor in state_dict.values(), so reversed state may change the flat dtype.
    Account for that here by comparing after casting to a common dtype.
    """
    _, state, meta = tinynet_state

    reversed_state = OrderedDict((k, state[k]) for k in reversed(list(state.keys())))

    flat1 = flatten_state_dict(state, meta)
    flat2 = flatten_state_dict(reversed_state, meta)

    # compare in common dtype
    assert torch.allclose(flat1.to(dtype=torch.float32), flat2.to(dtype=torch.float32))


def test_clone_state_dict_cpu(tinynet_state: TinyNetState) -> None:
    _, state, _ = tinynet_state

    clone = clone_state_dict(state, device=torch.device("cpu"))
    assert set(clone.keys()) == set(state.keys())

    for k, v in state.items():
        assert clone[k].device.type == "cpu"
        assert torch.allclose(v.cpu().to(dtype=torch.float32), clone[k].to(dtype=torch.float32))
        assert clone[k] is not v

    # Mutation should not leak back
    first_key = next(iter(state.keys()))
    clone[first_key].add_(1)
    assert not torch.allclose(
        clone[first_key].to(dtype=torch.float32),
        state[first_key].to(dtype=torch.float32),
    )


def test_model_reshaper_roundtrip(tinynet_state: TinyNetState) -> None:
    model, _, _ = tinynet_state

    reshaper = ModelReshaper()
    flat = reshaper.flatten(model)
    reshaper.unflatten_model(model, flat)
    flat2 = reshaper.flatten(model)

    assert torch.allclose(flat.to(dtype=torch.float32), flat2.to(dtype=torch.float32))


def test_unflatten_too_short_raises(tinynet_state: TinyNetState) -> None:
    """Make sure a shape mismatch blows up.

    unflatten_state_dict only fails when the reshaping fails (too short).
    Too-long flats are silently truncated to the prefix defined by metadata.
    """
    model, _, _ = tinynet_state
    reshaper = ModelReshaper()

    flat = reshaper.flatten(model)
    too_short = flat[:-1]  # definitely one element short

    with pytest.raises(RuntimeError):
        reshaper.unflatten_model(model, too_short)


# Property based tests -----------------------------------------------------------------------------


@st.composite
def tensor_shapes(draw: st.DrawFn) -> tuple[int, ...]:
    """Generate shapes up to rank 3, with side lengths in [0, 4]."""
    rank = draw(st.integers(min_value=0, max_value=3))
    if rank == 0:
        return ()
    dims = [draw(st.integers(min_value=0, max_value=4)) for _ in range(rank)]
    return tuple(dims)


@st.composite
def float_tensor(draw: st.DrawFn) -> Tensor:
    """Generate a float32 tensor with small rank and side lengths.

    Values are bounded to avoid inf/nan.
    """
    shape = draw(tensor_shapes())
    numel = 1
    for s in shape:
        numel *= s

    if numel == 0:
        return torch.empty(shape, dtype=torch.float32)

    vals = draw(
        st.lists(
            st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False),
            min_size=numel,
            max_size=numel,
        )
    )
    return torch.tensor(vals, dtype=torch.float32).view(shape)


@st.composite
def float_state_dicts(draw: st.DrawFn) -> StateDict:
    """Generate an OrderedDict[str, Tensor] with 1-5 float32 tensors.

    We keep these float-only for property tests, because mutation + int dtypes
    introduce truncation semantics that make "roundtrip after mutation" less
    meaningful to assert.
    """
    n_params = draw(st.integers(min_value=1, max_value=5))
    names = [f"p{i}" for i in range(n_params)]
    tensors = [draw(float_tensor()) for _ in range(n_params)]
    return OrderedDict((k, t) for k, t in zip(names, tensors))


@given(state=float_state_dicts())
@settings(max_examples=40)
def test_property_roundtrip(state: StateDict) -> None:
    meta = build_flat_metadata(state)
    flat = flatten_state_dict(state, meta)
    restored = unflatten_state_dict(flat, meta)

    assert set(state.keys()) == set(restored.keys())
    for k in state:
        assert state[k].dtype == torch.float32
        assert restored[k].dtype == torch.float32
        assert torch.allclose(state[k], restored[k])


@given(
    state=float_state_dicts(),
    shift=st.floats(min_value=-1.0, max_value=1.0),
)
@settings(max_examples=40)
def test_property_mutation_roundtrip(state: StateDict, shift: float) -> None:
    """Check roundtrip.

    If we mutate the flat tensor additively (for float dtypes), then
    unflatten + flatten should preserve that modification exactly.
    """
    meta = build_flat_metadata(state)
    flat = flatten_state_dict(state, meta)
    assert flat.dtype == torch.float32

    modified = flat + shift
    rebuilt = unflatten_state_dict(modified, meta)
    reflat = flatten_state_dict(rebuilt, meta)

    assert torch.allclose(modified, reflat)
