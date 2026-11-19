"""Read the frozen Sportloto archive and build next-draw examples."""

import csv
from itertools import islice
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from model import CONTEXT_LENGTH, NUMBER_COUNT, PREDICTION_SIZE


ARCHIVE_PATH = Path(__file__).resolve().parent / "data" / "sportloto_6x45_4853-19465.csv"
FIRST_DRAW = 4853
SPLIT_RANGES = {
    "train": (4873, 15087),
    "validation": (15088, 17276),
    "test": (17277, 19465),
}


class DrawDataset(Dataset[tuple[Tensor, Tensor]]):
    """An example contains the previous draws and the next draw to predict."""

    def __init__(self, draws: Tensor, start: int, stop: int) -> None:
        self.draws = draws
        self.start = start
        self.stop = stop

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        target = self.start + index
        return self.draws[target - CONTEXT_LENGTH : target], self.draws[target]


def load_datasets(
    path: Path = ARCHIVE_PATH, *, include_test: bool = True
) -> dict[str, DrawDataset]:
    """Encode each CSV row once and split examples by target draw number."""
    last_split = "test" if include_test else "validation"
    expected_count = SPLIT_RANGES[last_split][1] - FIRST_DRAW + 1
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader if include_test else islice(reader, expected_count):
            draw_id = int(row["draw_number"])
            expected_id = FIRST_DRAW + len(rows)
            if draw_id != expected_id:
                raise ValueError(f"Expected draw {expected_id}, got {draw_id}")

            numbers = {int(row[f"number_{i}"]) for i in range(1, PREDICTION_SIZE + 1)}
            if len(numbers) != PREDICTION_SIZE or not all(
                1 <= number <= NUMBER_COUNT for number in numbers
            ):
                raise ValueError(f"Invalid winning numbers in draw {draw_id}")

            rows.append([int(number in numbers) for number in range(1, NUMBER_COUNT + 1)])

    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} draws, got {len(rows)}")

    draws = torch.tensor(rows, dtype=torch.float32)
    return {
        name: DrawDataset(draws, first - FIRST_DRAW, last - FIRST_DRAW + 1)
        for name, (first, last) in SPLIT_RANGES.items()
        if include_test or name != "test"
    }
