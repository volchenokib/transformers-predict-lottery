"""Train the transformer on synthetic examples or the lottery archive."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import CONTEXT_LENGTH, NUMBER_COUNT, PREDICTION_SIZE, LotteryTransformer, select_numbers


def fit(
    model: nn.Module,
    train: DataLoader,
    validation: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
) -> tuple[int, float]:
    """Train on train batches and restore the best weights by validation loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    device = next(model.parameters()).device
    best_loss = float("inf")
    best_weights = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_values = 0.0, 0
        for inputs, targets in train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            total_loss = loss_fn(model(inputs), targets)
            (total_loss / targets.numel()).backward()
            optimizer.step()
            train_loss += total_loss.item()
            train_values += targets.numel()

        model.eval()
        validation_loss, validation_values = 0.0, 0
        with torch.inference_mode():
            for inputs, targets in validation:
                inputs, targets = inputs.to(device), targets.to(device)
                validation_loss += loss_fn(model(inputs), targets).item()
                validation_values += targets.numel()

        mean_loss = validation_loss / validation_values
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())
        print(f"epoch {epoch:02d} | train {train_loss / train_values:.4f} | validation {mean_loss:.4f}")

    if best_weights is None:
        raise RuntimeError("No epoch produced a finite validation loss")
    model.load_state_dict(best_weights)
    model.eval()
    return best_epoch, best_loss


def synthetic_examples(count: int, generator: torch.Generator) -> TensorDataset:
    """Generate independent histories whose target repeats their last draw."""
    ranks = torch.rand(count, CONTEXT_LENGTH, NUMBER_COUNT, generator=generator)
    indices = ranks.topk(PREDICTION_SIZE, dim=-1).indices
    histories = torch.zeros_like(ranks).scatter_(-1, indices, 1.0)
    return TensorDataset(histories, histories[:, -1, :].clone())


def check_synthetic() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(123)
    train = DataLoader(synthetic_examples(2048, generator), batch_size=64, shuffle=True)
    validation = DataLoader(synthetic_examples(256, generator), batch_size=64)
    holdout = DataLoader(synthetic_examples(256, generator), batch_size=64)
    model = LotteryTransformer()
    fit(model, train, validation, epochs=16, learning_rate=0.003)

    hits, examples = 0, 0
    with torch.inference_mode():
        for inputs, targets in holdout:
            picks = select_numbers(model(inputs))
            hits += int(targets.gather(1, picks - 1).sum().item())
            examples += len(targets)
    mean_hits = hits / examples
    print(f"Synthetic holdout: {mean_hits:.3f} / {PREDICTION_SIZE} mean hits ({examples} examples)")
    if mean_hits < 5.5:
        raise RuntimeError("The model did not reliably learn the synthetic rule")


def run_archive(checkpoint: Path) -> None:
    from dataset import load_datasets

    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    splits = load_datasets(include_test=False)
    train = splits["train"]
    validation = DataLoader(splits["validation"], batch_size=128)
    train_batches = DataLoader(train, batch_size=128, shuffle=True)
    # Include the initial 20 observed draws in the baseline's training history.
    counts = train.draws[:train.stop].sum(dim=0, keepdim=True)
    frequency_picks = select_numbers(counts)

    model = LotteryTransformer()
    best_epoch, best_loss = fit(model, train_batches, validation, epochs=12, learning_rate=0.001)
    model_hits = frequency_hits = 0
    with torch.inference_mode():
        for inputs, targets in validation:
            picks = select_numbers(model(inputs))
            model_hits += int(targets.gather(1, picks - 1).sum().item())
            frequency_hits += int(targets[:, frequency_picks[0] - 1].sum().item())

    examples = len(validation.dataset)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "validation_examples": examples,
        "validation_model_hits": model_hits,
        "validation_frequency_hits": frequency_hits,
        "frequency_picks": frequency_picks[0].tolist(),
    }, checkpoint)
    print(f"Validation ({examples} draws), mean hits: transformer {model_hits / examples:.4f}, "
          f"frequency {frequency_hits / examples:.4f}, random expectation 0.8000")
    print(f"Best epoch {best_epoch}, loss {best_loss:.6f}. Saved {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="Train on the archive without loading Test")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    args = parser.parse_args()
    if args.archive:
        run_archive(args.checkpoint)
    else:
        check_synthetic()
