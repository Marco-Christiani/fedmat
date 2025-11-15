from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from fedmat.vit_utils import get_amp_settings

if TYPE_CHECKING:
    from collections.abc import Iterable

    from transformers import ViTForImageClassification


@torch.inference_mode()
def evaluate(
    model: ViTForImageClassification,
    eval_batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    enable_bf16: bool,
) -> float:
    _ = model.eval()
    use_autocast, amp_dtype = get_amp_settings(device, enable_bf16)

    # device side counters to reduce syncs
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = torch.zeros((), device=device, dtype=torch.long)

    eval_iter = eval_batches
    if hasattr(eval_batches, "__len__"):
        eval_iter = tqdm(eval_batches, desc="Evaluating")

    for batch in eval_iter:
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

    accuracy_tensor = correct.float() / total.clamp_min(1).float()
    accuracy = float(accuracy_tensor.cpu())  # single sync
    return accuracy