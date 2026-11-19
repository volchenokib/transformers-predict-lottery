"""Evaluate the frozen pilot models on the final Test split."""

from math import comb
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import load_datasets
from model import NUMBER_COUNT, PREDICTION_SIZE, LotteryTransformer, select_numbers


SIMULATIONS = 10_000
SIMULATION_SEED = 2026
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best.pt"


def simulate_null(draw_count: int) -> torch.Tensor:
    """Sample totals under independent, uniformly random draws."""
    k = PREDICTION_SIZE
    weights = torch.tensor(
        [comb(k, hits) * comb(NUMBER_COUNT - k, k - hits) for hits in range(k + 1)],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(SIMULATION_SEED)
    totals = []
    for start in range(0, SIMULATIONS, 500):
        series_count = min(500, SIMULATIONS - start)
        hits = torch.multinomial(weights, series_count * draw_count, replacement=True, generator=generator)
        totals.append(hits.reshape(series_count, draw_count).sum(dim=1))
    return torch.cat(totals)


def evaluate() -> None:
    torch.set_num_threads(4)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = LotteryTransformer().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    frequency_indices = torch.tensor(checkpoint["frequency_picks"]) - 1
    test = load_datasets()["test"]
    histograms = {method: torch.zeros(PREDICTION_SIZE + 1, dtype=torch.long)
                  for method in ("transformer", "frequency")}

    with torch.inference_mode():
        for inputs, targets in DataLoader(test, batch_size=128):
            model_indices = select_numbers(model(inputs)) - 1
            for method, indices in (("transformer", model_indices), ("frequency", frequency_indices)):
                hits = targets[:, indices].sum(dim=1) if method == "frequency" else targets.gather(1, indices).sum(dim=1)
                histograms[method] += torch.bincount(hits.long(), minlength=PREDICTION_SIZE + 1)

    null_totals = simulate_null(len(test))
    random_mean = PREDICTION_SIZE ** 2 / NUMBER_COUNT
    print(f"Test draws: {len(test)}; random expectation: {random_mean:.6f}")
    for method, histogram in histograms.items():
        total = int(histogram @ torch.arange(PREDICTION_SIZE + 1))
        exceedances = int((null_totals >= total).sum())
        p_value = (1 + exceedances) / (SIMULATIONS + 1)
        mean = total / len(test)
        print(f"{method}: {total=}, {mean=:.6f}, delta={mean - random_mean:+.6f}, "
              f"{p_value=:.6f}, {exceedances=}, distribution={histogram.tolist()}")


if __name__ == "__main__":
    evaluate()
