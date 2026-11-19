"""Transformer for predicting the next Sportloto 6/45 draw."""

import torch
from torch import Tensor, nn


CONTEXT_LENGTH = 20
NUMBER_COUNT = 45
PREDICTION_SIZE = 6
MODEL_WIDTH = 64


class LotteryTransformer(nn.Module):
    """Map 20 previous multi-hot draws to one logit per lottery number."""

    def __init__(self) -> None:
        super().__init__()
        self.draw_projection = nn.Linear(NUMBER_COUNT, MODEL_WIDTH)
        self.position_embedding = nn.Parameter(torch.empty(CONTEXT_LENGTH, MODEL_WIDTH))
        nn.init.normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_WIDTH,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_head = nn.Linear(MODEL_WIDTH, NUMBER_COUNT)

    def forward(self, draws: Tensor) -> Tensor:
        """Accept float inputs of shape (batch, 20, 45), oldest draw first."""
        if draws.ndim != 3 or draws.shape[1:] != (CONTEXT_LENGTH, NUMBER_COUNT):
            raise ValueError(f"Expected shape (batch, 20, 45), got {tuple(draws.shape)}")

        tokens = self.draw_projection(draws) + self.position_embedding
        encoded_draws = self.encoder(tokens)
        return self.output_head(encoded_draws[:, -1, :])


def select_numbers(logits: Tensor) -> Tensor:
    """Return the top six numbers (1–45), favoring smaller numbers in ties."""
    if logits.ndim != 2 or logits.shape[1] != NUMBER_COUNT:
        raise ValueError(f"Expected shape (batch, 45), got {tuple(logits.shape)}")

    ranked = torch.argsort(logits, dim=-1, descending=True, stable=True)
    return ranked[:, :PREDICTION_SIZE] + 1
