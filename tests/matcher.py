import copy
import unittest

import torch
from fedmat.utils import create_vit_classifier, get_amp_settings, set_seed
from fedmat.matching import GreedyMatcher as GM, HungarianMatcher as HM
from transformers import ViTForImageClassification

class TestMatchers(unittest.TestCase):
    def test_hungarian_matcher_against_greedy(self):
        "We are testing the shape of the output of Hungarian against that of Greedy to ensure that they match"
        set_seed(42)
        client_models = [
                create_vit_classifier(
                    model_name="google/vit-base-patch16-224-in21k",
                    num_labels=10,
                    use_pretrained=False,
        ).cuda() for _ in range(10)]
        for layer_idx in range(len(client_models[0].vit.encoder.layer)):
            client_layers: list[ViTLayer] = [model.vit.encoder.layer[layer_idx] for model in client_models]
            hps = HM.match(client_layers)
            gps = GM.match(client_layers)
            assert len(hps) == len(gps)
            for hp, gp in zip(hps, gps):
                #assert torch.equal(hp, gp), f"{hp.numpy()}, {gp.numpy()}"
                assert hp.shape == gp.shape
                assert torch.equal(torch.unique(hp).sort().values, torch.unique(gp).sort().values)

if __name__ == '__main__':
    unittest.main()
