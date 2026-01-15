from __future__ import annotations

from pathlib import Path
import torch
import json
from transformers import PreTrainedModel


def save_checkpoint(
    ckpt_dir: str | Path,
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: any,
    epoch: int,
    best_val_loss: float,
    val_metrics: dict[str, float],
    ) -> None:
    """Saves a training checkpoint to `ckpt_dir`.

    Layout:
      ckpt_dir/
        model/              (HF save_pretrained)
        tokenizer/          (HF save_pretrained, optional)
        trainer_state.pt    (torch.save dict)
    
    Args:
        ckpt_dir: Directory to save the checkpoint.
        model: Model to save.
        optimizer: Optimizer whose state to save.
        scheduler: Scheduler whose state to save.
        epoch: Current epoch number.
        best_val_loss: Best validation loss so far.
        val_metrics: Validation metrics dictionary.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save HF model weights/config
    model_dir = ckpt_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)

    # Save optimizer/scheduler + training state
    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state, ckpt_dir / "trainer_state.pt")

    # Save validation metrics for all epochs
    with open(ckpt_dir / "validation_metrics.jsonl", "a", encoding="utf-8") as f:
        json.dump({"epoch": epoch, **val_metrics}, f)
        f.write("\n")


def load_checkpoint(
    ckpt_dir: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler: any,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Loads trainer_state.pt and restore optimizer/scheduler if provided.
    Note: model weights are loaded via HF from_pretrained usually; here we restore optimizer/scheduler + metadata.

    Args:
        ckpt_dir: Directory of the checkpoint.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        map_location: Device mapping for loading.

    Returns:
        state dict containing epoch/global_step/best_val_loss/cfg...
    """
    state = torch.load(
        Path(ckpt_dir) / "trainer_state.pt",
        map_location=map_location,
    )

    optimizer.load_state_dict(state.pop("optimizer_state_dict"))
    scheduler.load_state_dict(state.pop("scheduler_state_dict"))

    return state
