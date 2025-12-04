"""Evaluation utilities for model assessment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from fedmat.utils import get_amp_settings

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import ViTForImageClassification

    from fedmat.data import Batch


@torch.inference_mode()
def evaluate(
    model: ViTForImageClassification,
    dataloader: DataLoader[Batch],
    device: torch.device,
    enable_bf16: bool,
) -> tuple[float, torch.Tensor]:
    """Evaluate model on a dataset and return accuracy and confusion matrix.

    Parameters
    ----------
    model : ViTForImageClassification
        Vision Transformer model to evaluate
    dataloader : DataLoader[Batch]
        DataLoader providing batches of data
    device : torch.device
        Device to run evaluation on
    enable_bf16 : bool
        Whether to use bfloat16 precision

    Returns
    -------
    tuple[float, torch.Tensor]
        Tuple of (accuracy, confusion_matrix)
    """
    _ = model.eval()
    use_autocast, amp_dtype = get_amp_settings(device, enable_bf16)

    correct = torch.zeros((), device=device, dtype=torch.long)
    total = torch.zeros((), device=device, dtype=torch.long)
    num_labels = model.config.num_labels
    conf = torch.zeros((num_labels, num_labels), device=device, dtype=torch.long)

    for batch in tqdm(dataloader, desc="Evaluating"):
        labels = batch["labels"].to(device=model.device)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_autocast,
        ):
            pixel_values = batch["pixel_values"].to(device=model.device, dtype=model.dtype)
            outputs = model(pixel_values=pixel_values)

        preds = outputs.logits.argmax(dim=-1)  # [B]
        correct += (preds == labels).sum()
        total += labels.size(0)
        idx = labels * num_labels + preds
        conf.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))

    accuracy_tensor = correct.float() / total.clamp_min(1).float()
    accuracy = float(accuracy_tensor.cpu())  # single sync
    return accuracy, conf
