from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fedmat.matching import GreedyMatcher, HungarianMatcher
from fedmat.utils import (
    as_vit_layer,
    create_vit_classifier,
    default_device,
    set_seed,
)

if TYPE_CHECKING:
    from transformers import ViTForImageClassification
    from transformers.models.vit.modeling_vit import ViTLayer


# Fixtures ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def device() -> torch.device:
    return default_device()


@pytest.fixture(scope="module")
def client_models(device: torch.device) -> list[ViTForImageClassification]:
    """Create a small batch of models."""
    set_seed(42)
    models = [
        create_vit_classifier(
            model_name="google/vit-base-patch16-224-in21k",
            num_labels=10,
            use_pretrained=False,
        ).to(device)  # type: ignore
        for _ in range(10)
    ]
    return models


@pytest.fixture
def encoder_layers(client_models: list[ViTForImageClassification]) -> list[list[ViTLayer]]:
    """Yield layers per index across all clients."""
    layer_count = len(client_models[0].vit.encoder.layer)
    all_layers: list[list[ViTLayer]] = []

    for idx in range(layer_count):
        layers_for_idx: list[ViTLayer] = [
            as_vit_layer(model.vit.encoder.layer[idx])
            for model in client_models
        ]
        all_layers.append(layers_for_idx)

    return all_layers


# Tests ------------------------------------------------------------------------

def test_matchers_return_valid_permutations(encoder_layers: list[list[ViTLayer]], device: torch.device) -> None:
    """Check matchers return valid permutations.

    For each encoder layer index:
      - Hungarian and Greedy produce same shape
      - Each permutation is a true permutation: contains exactly {0..H-1}
      - Both methods return a list of length == num_clients
    """
    for layer_group in encoder_layers:
        greedy = GreedyMatcher.match(layer_group)
        hungarian = HungarianMatcher.match(layer_group)

        assert len(greedy) == len(hungarian)

        for g_perm, h_perm in zip(greedy, hungarian):
            assert g_perm.shape == h_perm.shape

            # Both must be valid index permutations
            # TODO: We do not insist they match each other because algorithms differ and we
            #  should probably craft inputs for that reason.
            g_sorted = torch.sort(torch.unique(g_perm)).values
            h_sorted = torch.sort(torch.unique(h_perm)).values
            expected = torch.arange(len(g_sorted), dtype=torch.long, device=device)

            assert torch.equal(g_sorted, expected)
            assert torch.equal(h_sorted, expected)


def test_matchers_are_deterministic(encoder_layers: list[list[ViTLayer]]) -> None:
    """Running match twice with same inputs should produce identical results."""
    for layer_group in encoder_layers:
        p1_g = GreedyMatcher.match(layer_group)
        p2_g = GreedyMatcher.match(layer_group)
        p1_h = HungarianMatcher.match(layer_group)
        p2_h = HungarianMatcher.match(layer_group)

        for a, b in zip(p1_g, p2_g):
            assert torch.equal(a, b)

        for a, b in zip(p1_h, p2_h):
            assert torch.equal(a, b)


def test_matchers_use_same_head_count(encoder_layers: list[list[ViTLayer]]) -> None:
    """Cross matcher consistency test, WIP since failable.

    Sanity check: both matchers should infer identical head count H for each layer.
    This does not require that the permutations match, should be adapted...
    """
    # TODO: We do not insist they match each other because algorithms differ and we
    #  should probably craft inputs for that reason.
    for layer_group in encoder_layers:
        greedy = GreedyMatcher.match(layer_group)
        hungarian = HungarianMatcher.match(layer_group)

        assert greedy[0].numel() == hungarian[0].numel()
