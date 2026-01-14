from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.utils.config import load_yaml_config
from src.utils.logging import get_logger, make_run_dir
from src.utils.seed import set_seed

from src.models.load import load_model
from src.data.dataloaders import prepare_dataloader


def train_step(
        model: AutoModelForSequenceClassification,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        device: torch.device,
        optimizer,
        lr_scheduler,
        ) ->tuple[float, float]:
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
    
    Returns:
        A tuple containing the average training loss and average validation loss.
    """
    # Initialization
    epoch_loss = 0.0
    val_epoch_loss = 0.0
    num_training_steps = len(train_dataloader)
    num_validation_steps = len(validation_dataloader)

    # Training loop
    model.train()
    for batch in train_dataloader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        epoch_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        for batch in validation_dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            val_epoch_loss += loss.item()
    
    return epoch_loss / num_training_steps, val_epoch_loss / num_validation_steps


def train(
    model: AutoModelForSequenceClassification,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    optimizer,
    lr_scheduler,
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
    """
    for epoch in range(epochs):
        train_loss, val_loss = train_step(
            model,
            train_dataloader,
            validation_dataloader,
            device,
            optimizer,
            lr_scheduler,
        )


def main() -> None:
    """High-level evaluation entry point."""
    # Configuration and setup
    cfg = load_yaml_config()
    set_seed(cfg['random']['seed'])
    run_name = (
        "train_"
        f"{cfg['data']['dataset_name'].split('/')[-1]}_"
        f"{cfg['model']['pretrained_model_name'].replace('/', '-')}"
    )
    run_dir = make_run_dir(base_dir='results', run_name=run_name)
    logger = get_logger(name=run_name, log_file=run_dir / 'run.log')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting training...")

    # Instantiate the tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['pretrained_model_name'])

    # Prepare the dataloaders
    logger.info("Preparing dataloaders...")
    train_dataloader, labels = prepare_dataloader(
        dataset_name=cfg['data']['dataset_name'],
        split=cfg['train']['split'],
        tokenizer=tokenizer,
        batch_size=cfg['train']['batch_size'],
        max_length=cfg['data']['max_length'],
        shuffle=True,
        keep_text=cfg['train']['keep_text'],
        num_proc=cfg['data']['num_proc'],
    )
    validation_dataloader, _ = prepare_dataloader(
        dataset_name=cfg['data']['dataset_name'],
        split=cfg['validation']['split'],
        tokenizer=tokenizer,
        batch_size=cfg['validation']['batch_size'],
        max_length=cfg['data']['max_length'],
        shuffle=False,
        keep_text=cfg['validation']['keep_text'],
        num_proc=cfg['data']['num_proc'],
    )
    num_labels = len(labels)

    # Load the model
    logger.info("Loading model...")
    model = load_model(
        model_name=cfg['model']['pretrained_model_name'],
        num_labels=num_labels,
        device=device,
    )

    # Prepare the optimizer, learning rate scheduler, and loss function
    optimizer = AdamW(model.parameters(), lr=cfg['train']['learning_rate'])
    lr_sched = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg['train']['warmup_steps'],
        num_training_steps=cfg['train']['epochs'] * len(train_dataloader),
    )

    # Training
    train(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        device=device,
        epochs=cfg['train']['epochs'],
        optimizer=optimizer,
        lr_scheduler=lr_sched,
    )


if __name__ == "__main__":
    main()
