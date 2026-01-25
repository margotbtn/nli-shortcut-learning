# src/training/train.py
from __future__ import annotations

from typing import Any
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.training.eval import evaluate
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger, make_run_dir
from src.utils.seed import set_seed
from src.utils.typing import InputView, Checkpoint
from src.models.load import load_model
from src.models.checkpoints import load_checkpoint, save_checkpoint
from src.data.dataloaders import prepare_dataloader


def train_step(
        model: AutoModelForSequenceClassification,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        logger: logging.Logger,
        epoch: int,
        checkpoint_every_n_steps: int | None = None,
        run_dir: str | Path | None = None,
        state: dict[str, Any] | None = None,
        skip_steps: int = 0,
        ) ->tuple[float, float, dict[str, float]]:
    """Trains the model for one epoch.

    General logic:
        1. Loop over training batches: forward pass, compute loss, backward pass, optimizer step.
        2. Loop over validation batches: forward pass, compute loss.

    Args:
        model: The sequence classification model to train.
        train_dataloader: DataLoader for the training dataset.
        validation_dataloader: DataLoader for the validation dataset.
        device: Device to run the training on.
        optimizer: Optimizer for model parameters.
        lr_scheduler: Learning rate scheduler.
        logger: Logger for logging training progress.
        epoch: Current epoch index.
        checkpoint_every_n_steps: Save latest checkpoint every N steps if provided.
        run_dir: Directory of the training run (required for step checkpoints).
        state: Training state dictionary to update.
        skip_steps: Number of initial steps to skip (resume mid-epoch).
    
    Returns:
        Average training loss for the epoch.
        Average validation loss for the epoch.
        Validation metrics dictionary over the validation dataset.
    """
    # Initialization
    epoch_loss = 0.0
    num_training_steps = max(1, len(train_dataloader) - skip_steps)

    # Training loop
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        if step < skip_steps:
            continue
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        epoch_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if state is not None:
            state["global_step"] += 1
            state["step_in_epoch"] = step + 1

            if (
                checkpoint_every_n_steps
                and run_dir is not None
                and state["global_step"] % checkpoint_every_n_steps == 0
            ):
                logger.info("Saving intermediate checkpoint.")
                save_checkpoint(
                    ckpt_dir=Path(run_dir) / 'checkpoints' / 'latest',
                    model=model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    epoch=epoch,
                    best_val_loss=state["best_val_loss"],
                    val_metrics={},
                    global_step=state["global_step"],
                    step_in_epoch=state["step_in_epoch"],
                )
    
    # Validation loop
    val_metrics, _, val_loss = evaluate(
        model=model,
        dataloader=validation_dataloader,
        device=device,
        num_labels=len(validation_dataloader.dataset.features["labels"].names),
        logger=logger,
    )
    
    train_loss = epoch_loss / num_training_steps

    return train_loss, val_loss, val_metrics


def train(
    model: AutoModelForSequenceClassification,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    logger: logging.Logger,
    run_dir: str | Path,
    saved_state: dict[str, Any] | None = None,
    checkpoint_every_n_steps: int | None = None,
    seed: int | None = None,
    train_generator: torch.Generator | None = None,
    ) -> None:
    """High-level training loop.
    
    Args:
        model: The sequence classification model to train.
        train_dataloader: DataLoader for the training dataset.
        validation_dataloader: DataLoader for the validation dataset.
        device: Device to run the training on.
        epochs: Number of training epochs.
        optimizer: Optimizer for model parameters.
        lr_scheduler: Learning rate scheduler.
        logger: Logger for logging training progress.
        run_dir: Directory of the training run.
        saved_state: Optional training state to resume from.
        checkpoint_every_n_steps: Save latest checkpoint every N steps if provided.
        seed: Base seed for deterministic shuffling.
        train_generator: DataLoader generator used for shuffling.
    """
    # Initialize training state
    state = {
        "epoch": 0,
        "best_val_loss": float("inf"),
        "global_step": 0,
        "step_in_epoch": 0,
    }
    state.update(saved_state or {})
    state["global_step"] = state.get("global_step") or 0
    state["step_in_epoch"] = state.get("step_in_epoch") or 0

    if saved_state:
        load_checkpoint(
            ckpt_dir=Path(run_dir) / 'checkpoints' / 'latest',
            optimizer=optimizer,
            scheduler=lr_scheduler,
            map_location=device,
        )

    for epoch in range(state['epoch'], epochs):
        if train_generator is not None and seed is not None:
            train_generator.manual_seed(seed + epoch)

        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"Current state: {state}")

        skip_steps = state["step_in_epoch"] if epoch == state["epoch"] else 0

        # Perform one training epoch
        train_loss, val_loss, val_metrics = train_step(
            model=model,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=logger,
            epoch=epoch,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            run_dir=run_dir,
            state=state,
            skip_steps=skip_steps,
        )

        # If better validation loss, save the best checkpoint
        if val_loss < state['best_val_loss']:
            logger.info("New best validation loss. Saving best checkpoint.")
            state['best_val_loss'] = val_loss
            save_checkpoint(
                ckpt_dir=Path(run_dir) / 'checkpoints' / 'best',
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                epoch=epoch + 1,
                best_val_loss=state['best_val_loss'],
                val_metrics=val_metrics,
                global_step=state["global_step"],
                step_in_epoch=0,
            )
        
        # Save the latest checkpoint
        logger.info(f"Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        logger.info("Saving latest checkpoint.")
        save_checkpoint(
            ckpt_dir=Path(run_dir) / 'checkpoints' / 'latest',
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            epoch=epoch + 1,
            best_val_loss=state['best_val_loss'],
            val_metrics=val_metrics,
            global_step=state["global_step"],
            step_in_epoch=0,
        )

        state['epoch'] = epoch + 1
        state["step_in_epoch"] = 0


def main(
    input_view: InputView = "pair",
    checkpoint: Checkpoint = "pretrained",
    run_dir: str | None = None,
) -> None:
    """High-level training entry point."""
    # Get the configuration
    overrides = {
        ('data', 'input_view'): input_view,
        ("train", "checkpoint"): checkpoint,
        ("train", "run_dir"): run_dir,
    }
    cfg = load_yaml_config('src/config.yaml', overrides)

    # Set the seed
    set_seed(cfg['random']['seed'])

    # Set run_name and run_dir
    if cfg['train']['checkpoint'] == 'pretrained':
        run_name = (
            "train_"
            f"{input_view}_"
            f"{cfg['data']['dataset_name'].split('/')[-1]}_"
            f"{cfg['model']['pretrained_model_name'].replace('/', '-')}"
        )
        run_dir = make_run_dir(base_dir='results', run_name=run_name)
    else:
        if not Path(cfg['train']['run_dir']).exists():
            raise ValueError(f"Provided run_dir '{cfg['train']['run_dir']}' does not exist.")
        run_dir = Path(cfg['train']['run_dir'])
        run_name = run_dir.name
    
    # Set up the logger
    logger = get_logger(name=run_name, log_file=run_dir / 'run.log')

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting training...")

    # Instantiate the tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['pretrained_model_name'])

    # Prepare the dataloaders
    logger.info("Preparing dataloaders...")
    train_generator = torch.Generator()
    train_dataloader, labels = prepare_dataloader(
        dataset_name=cfg['data']['dataset_name'],
        split=cfg['train']['split'],
        input_view=cfg['data']['input_view'],
        tokenizer=tokenizer,
        batch_size=cfg['train']['batch_size'],
        max_length=cfg['data']['max_length'],
        shuffle=True,
        keep_text=cfg['train']['keep_text'],
        num_proc=cfg['data']['num_proc'],
        num_workers=cfg['data']['num_workers'],
        generator=train_generator,
    )
    validation_dataloader, _ = prepare_dataloader(
        dataset_name=cfg['data']['dataset_name'],
        split=cfg['validation']['split'],
        input_view=cfg['data']['input_view'],
        tokenizer=tokenizer,
        batch_size=cfg['validation']['batch_size'],
        max_length=cfg['data']['max_length'],
        shuffle=False,
        keep_text=cfg['validation']['keep_text'],
        num_proc=cfg['data']['num_proc'],
        num_workers=cfg['data']['num_workers'],
    )
    num_labels = len(labels)

    # Load the model
    logger.info("Loading model...")
    model = load_model(
        model_name=cfg['model']['pretrained_model_name'],
        num_labels=num_labels,
        device=device,
        checkpoint=cfg['train']['checkpoint'],
        run_dir=run_dir,
    )

    # Prepare the optimizer, learning rate scheduler, and loss function
    logger.info("Preparing optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=cfg['train']['learning_rate'])
    lr_sched = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg['train']['warmup_step'],
        num_training_steps=cfg['train']['epochs'] * len(train_dataloader),
    )

    # Load saved training state if resuming
    saved_state = dict()
    if cfg['train']['checkpoint'] != 'pretrained':
        logger.info("Loading saved training state...")
        saved_state = load_checkpoint(
            ckpt_dir=Path(cfg['train']['run_dir']) / 'checkpoints' / 'latest',
            optimizer=optimizer,
            scheduler=lr_sched,
            map_location=device,
        )
    else:
        saved_state = None

    # Training
    logger.info("Beginning training loop...")
    train(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        device=device,
        epochs=cfg['train']['epochs'],
        optimizer=optimizer,
        lr_scheduler=lr_sched,
        logger=logger,
        saved_state=saved_state,
        run_dir=run_dir,
        checkpoint_every_n_steps=cfg['train']['checkpoint_every_n_steps'],
        seed=cfg['random']['seed'],
        train_generator=train_generator,
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument(
        "--input_view",
        choices=["pair", "hypothesis_only"],
        default="pair",
        help="Train and evaluate the model either on both sentences or only the hypothesis.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["latest", "pretrained"],
        default="pretrained",
        help="Checkpoint to start from (train only supports latest/pretrained).",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Run directory (required when checkpoint is not pretrained).",
    )
    args = parser.parse_args()
    main(args.input_view, args.checkpoint, args.run_dir)