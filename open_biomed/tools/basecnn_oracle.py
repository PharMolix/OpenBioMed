"""BaseCNN fitness oracle, vendored from the GGS framework.

The architecture and forward pass are reproduced from the public GGS repository
(https://github.com/kirjner/GGS, `ggs/models/predictors.py`) so that the AAV/GFP
oracle checkpoints shipped with the mutation-design skills can be loaded and
queried directly. The original tool stubbed ``_score_sequence`` with a placeholder
because it did not have this class definition; with the real forward the trained
checkpoints score their own training data at Spearman ~0.87-0.95.

Reference validation (2026-06-25), real BaseCNN forward vs GGS ``ground_truth.csv``:
  - AAV: Spearman 0.9487 / Pearson 0.9435
  - GFP: Spearman 0.8731 / Pearson 0.9612
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


class LengthMaxPool1D(nn.Module):
    """Per-position Linear + activation, then max-pool over the length axis.

    This is the embedding block of ``BaseCNN``. The ordering matters: the Linear
    is applied to every position first, THEN max-pool collapses the length axis.
    Pooling before the Linear, or applying the decoder per-position, both give
    Spearman ~0 — only this exact ordering matches the trained checkpoints.
    """

    def __init__(self, in_dim: int, out_dim: int, linear: bool = True, activation: str = "relu") -> None:
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)
        if activation == "relu":
            self.act_fn = lambda x: F.relu(x)
        elif activation == "swish":
            self.act_fn = lambda x: x * torch.sigmoid(100.0 * x)
        elif activation == "softplus":
            self.act_fn = nn.Softplus()
        elif activation == "sigmoid":
            self.act_fn = nn.Sigmoid()
        elif activation == "leakyrelu":
            self.act_fn = nn.LeakyReLU()
        else:
            raise NotImplementedError(f"activation {activation} not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, in_dim)
        if self.linear:
            x = self.act_fn(self.layer(x))        # (B, L, out_dim)
        x = torch.max(x, dim=1)[0]                # (B, out_dim)
        return x


class BaseCNN(nn.Module):
    """1D-CNN protein fitness predictor (the GGS "CNN" oracle).

    Forward expects integer-tokenized sequences of shape (B, L). With
    ``make_one_hot=True`` (default) the one-hot expansion is done internally,
    matching how the checkpoints were trained.
    """

    def __init__(
        self,
        n_tokens: int = 20,
        kernel_size: int = 5,
        input_size: int = 256,
        dropout: float = 0.0,
        make_one_hot: bool = True,
        activation: str = "relu",
        linear: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(
            linear=linear,
            in_dim=input_size,
            out_dim=input_size * 2,
            activation=activation,
        )
        self.decoder = nn.Linear(input_size * 2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self._make_one_hot = make_one_hot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: integer tokens (B, L)
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()              # (B, n_tokens, L)
        x = self.encoder(x).permute(0, 2, 1)        # (B, L', input_size)
        x = self.dropout(x)
        x = self.embedding(x)                       # (B, input_size*2)
        output = self.decoder(x).squeeze(1)         # (B,)
        return output


def load_oracle(model_path: str, config_path: str, device: str = "cpu") -> "BaseCNN":
    """Build a BaseCNN from a GGS config and load a Lightning checkpoint into it.

    The checkpoint ``state_dict`` keys are prefixed with ``predictor.`` (the
    pytorch-lightning module attribute name); strip it before ``load_state_dict``.
    """
    with open(config_path, "r") as f:
        cfg = OmegaConf.load(f)
    predictor_cfg = cfg.model.predictor
    model = BaseCNN(**predictor_cfg)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("predictor.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def encode_sequences(sequences: List[str], alphabet: str) -> torch.Tensor:
    """Integer-encode a batch of equal-length sequences under the given alphabet.

    The alphabet order MUST match the one the checkpoint was trained with
    (config ``data.alphabet``); reordering it silently destroys all rank signal.
    """
    a_to_i = {a: i for i, a in enumerate(alphabet)}
    encoded = []
    for seq in sequences:
        encoded.append([a_to_i[a] for a in seq])
    return torch.tensor(encoded, dtype=torch.long)


def score_sequences(model: "BaseCNN", sequences: List[str], alphabet: str, device: str = "cpu") -> List[float]:
    """Score a batch of sequences with a loaded BaseCNN oracle.

    Returns one float fitness score per input sequence, in order.
    """
    if not sequences:
        return []
    tokens = encode_sequences(sequences, alphabet).to(device)
    with torch.no_grad():
        out = model(tokens)
    return out.detach().cpu().tolist()


def get_alphabet(config_path: str) -> str:
    """Read the training alphabet from a GGS config (used for integer encoding)."""
    with open(config_path, "r") as f:
        cfg = OmegaConf.load(f)
    return cfg.data.alphabet
