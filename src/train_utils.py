from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    return images, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochResult:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in loader:
        images, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += labels.size(0)

    return EpochResult(loss=total_loss / total_seen, accuracy=total_correct / total_seen)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in loader:
        images, labels = _move_batch(batch, device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += labels.size(0)

    return EpochResult(loss=total_loss / total_seen, accuracy=total_correct / total_seen)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    labels_out: list[int] = []
    probabilities_out: list[float] = []
    paths_out: list[str] = []

    for batch in loader:
        images, labels = _move_batch(batch, device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        labels_out.extend(labels.cpu().numpy().tolist())
        probabilities_out.extend(probabilities.cpu().numpy().tolist())
        paths_out.extend(batch["path"])

    return np.asarray(labels_out), np.asarray(probabilities_out), paths_out

