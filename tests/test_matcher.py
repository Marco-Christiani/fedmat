import unittest
from typing import TYPE_CHECKING

import torch

from fedmat.matching import GreedyMatcher, HungarianMatcher
from fedmat.utils import as_vit_layer, create_vit_classifier, default_device, set_seed

if TYPE_CHECKING:
    from transformers.models.vit.modeling_vit import ViTLayer


class TestMatchers(unittest.TestCase):
    def test_hungarian_matcher_against_greedy(self) -> None:
        """We are testing the shape of the output of Hungarian against that of Greedy to ensure that they match."""
        set_seed(42)
        device = default_device()
        client_models = [
            create_vit_classifier(
                model_name="google/vit-base-patch16-224-in21k", num_labels=10, use_pretrained=False
            ).to(device=device)  # type: ignore
            for _ in range(10)
        ]
        for layer_idx in range(len(client_models[0].vit.encoder.layer)):
            client_layers: list[ViTLayer] = [
                as_vit_layer(model.vit.encoder.layer[layer_idx]) for model in client_models
            ]
            hps = HungarianMatcher.match(client_layers)
            gps = GreedyMatcher.match(client_layers)
            assert len(hps) == len(gps)
            for hp, gp in zip(hps, gps):
                # assert torch.equal(hp, gp), f"{hp.numpy()}, {gp.numpy()}"
                assert hp.shape == gp.shape
                assert torch.equal(torch.unique(hp).sort().values, torch.unique(gp).sort().values)


if __name__ == "__main__":
    unittest.main()
