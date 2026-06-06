from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DRAWING_TYPES = ("spiral", "wave")
CLASS_TO_IDX = {"healthy": 0, "parkinson": 1}
IDX_TO_CLASS = {0: "Healthy", 1: "Parkinson's"}


@dataclass(frozen=True)
class DrawingRecord:
    path: Path
    label: str
    label_idx: int
    drawing_type: str
    split: str


def _normalize_token(token: str) -> str:
    return token.lower().replace("-", "_").replace(" ", "_").replace("'", "")


def _infer_label(parts: Iterable[str]) -> str | None:
    normalized = [_normalize_token(part) for part in parts]
    for part in normalized:
        if part in {"healthy", "control", "controls", "normal"}:
            return "healthy"
        if part in {"parkinson", "parkinsons", "parkinson_disease", "pd"}:
            return "parkinson"
    return None


def _infer_drawing_type(parts: Iterable[str]) -> str | None:
    normalized = [_normalize_token(part) for part in parts]
    for drawing_type in DRAWING_TYPES:
        if drawing_type in normalized:
            return drawing_type
    return None


def _infer_split(parts: Iterable[str]) -> str:
    normalized = [_normalize_token(part) for part in parts]
    if any(part in {"train", "training"} for part in normalized):
        return "train"
    if any(part in {"test", "testing"} for part in normalized):
        return "test"
    if any(part in {"valid", "validation", "val"} for part in normalized):
        return "val"
    return "unsplit"


def discover_image_records(
    data_dir: str | Path,
    split: str | None = None,
    drawing_type: str | None = None,
) -> list[DrawingRecord]:
    """Discover labeled drawing images from the Kaggle directory tree.

    The Kaggle dataset is usually nested as:
    drawings/{spiral,wave}/{training,testing}/{healthy,parkinson}/image.png

    This scanner intentionally infers metadata from all path parts so it still
    works if the dataset is copied with one extra parent folder.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        return []

    requested_split = split.lower() if split else None
    requested_drawing_type = drawing_type.lower() if drawing_type else None
    records: list[DrawingRecord] = []

    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parts = image_path.relative_to(root).parts
        inferred_label = _infer_label(parts)
        inferred_drawing_type = _infer_drawing_type(parts)
        inferred_split = _infer_split(parts)

        if inferred_label is None or inferred_drawing_type is None:
            continue
        if requested_split and inferred_split != requested_split:
            continue
        if requested_drawing_type and requested_drawing_type != "all":
            if inferred_drawing_type != requested_drawing_type:
                continue

        records.append(
            DrawingRecord(
                path=image_path,
                label=inferred_label,
                label_idx=CLASS_TO_IDX[inferred_label],
                drawing_type=inferred_drawing_type,
                split=inferred_split,
            )
        )

    return records


class ParkinsonDrawingDataset(Dataset):
    def __init__(self, records: list[DrawingRecord], transform=None):
        if not records:
            raise ValueError("No drawing records were provided.")
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(record.label_idx, dtype=torch.long),
            "drawing_type": record.drawing_type,
            "split": record.split,
            "path": str(record.path),
        }


def get_train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=8, fill=255),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_dataloaders(
    data_dir: str | Path,
    drawing_type: str = "all",
    image_size: int = 224,
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    train_records = discover_image_records(data_dir, split="train", drawing_type=drawing_type)
    test_records = discover_image_records(data_dir, split="test", drawing_type=drawing_type)

    if not train_records:
        all_records = discover_image_records(data_dir, drawing_type=drawing_type)
        if not all_records:
            raise FileNotFoundError(
                f"No labeled images found under {Path(data_dir).resolve()}. "
                "Run `python data.py` first, or point --data-dir at the downloaded dataset."
            )
        train_records = all_records

    if len(train_records) > 1 and val_split > 0:
        val_size = max(1, int(len(train_records) * val_split))
        val_size = min(val_size, len(train_records) - 1)
    else:
        val_size = 0

    if val_size:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_records), generator=generator).tolist()
        val_indices = set(indices[:val_size])
        split_train_records = [record for idx, record in enumerate(train_records) if idx not in val_indices]
        split_val_records = [record for idx, record in enumerate(train_records) if idx in val_indices]
    else:
        split_train_records = train_records
        split_val_records = []

    train_dataset = ParkinsonDrawingDataset(split_train_records, transform=get_train_transform(image_size))
    val_dataset = (
        ParkinsonDrawingDataset(split_val_records, transform=get_eval_transform(image_size))
        if split_val_records
        else None
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_dataset is not None
        else None
    )

    test_loader = None
    if test_records:
        test_dataset = ParkinsonDrawingDataset(test_records, transform=get_eval_transform(image_size))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def summarize_records(records: list[DrawingRecord]) -> list[dict[str, object]]:
    counts = Counter((record.split, record.drawing_type, record.label) for record in records)
    rows: list[dict[str, object]] = []
    for (split, drawing_type, label), count in sorted(counts.items()):
        rows.append(
            {
                "split": split,
                "drawing_type": drawing_type,
                "class": IDX_TO_CLASS[CLASS_TO_IDX[label]],
                "count": count,
            }
        )
    return rows


def find_examples(records: list[DrawingRecord], drawing_type: str, label: str, limit: int = 6) -> list[Path]:
    label_key = label.lower().replace("'", "").replace(" ", "")
    if label_key.startswith("parkinson"):
        label_key = "parkinson"
    elif label_key.startswith("healthy"):
        label_key = "healthy"

    return [
        record.path
        for record in records
        if record.drawing_type == drawing_type and record.label == label_key
    ][:limit]
