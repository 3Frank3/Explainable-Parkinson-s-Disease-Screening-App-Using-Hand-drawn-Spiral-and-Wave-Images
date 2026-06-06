from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.dataset import discover_image_records, summarize_records


DATASET_SLUG = "kmader/parkinsons-drawings"
DEFAULT_TARGET_DIR = Path("data/raw/parkinsons-drawings")


def download_dataset(target_dir: Path = DEFAULT_TARGET_DIR) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Install dependencies with "
            "`pip install -r requirements.txt` and run this script again."
        ) from exc

    downloaded_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    target_dir = target_dir.expanduser().resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(downloaded_path, target_dir, dirs_exist_ok=True)
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Parkinson's drawings Kaggle dataset.")
    parser.add_argument(
        "--target-dir",
        default=str(DEFAULT_TARGET_DIR),
        help="Directory where the dataset should be copied for this project.",
    )
    args = parser.parse_args()

    target_dir = download_dataset(Path(args.target_dir))
    records = discover_image_records(target_dir)
    print(f"Dataset copied to: {target_dir}")
    print(f"Discovered labeled images: {len(records)}")

    for row in summarize_records(records):
        print(
            f"{row['split']:>7} | {row['drawing_type']:<6} | "
            f"{row['class']:<12} | {row['count']}"
        )


if __name__ == "__main__":
    main()
