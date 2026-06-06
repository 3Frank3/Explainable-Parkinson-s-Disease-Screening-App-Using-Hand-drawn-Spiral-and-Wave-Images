from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.dataset import build_dataloaders
from src.evaluate import compute_binary_metrics, load_metrics, save_metrics
from src.model import create_model, save_checkpoint
from src.train_utils import collect_predictions, evaluate_epoch, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Parkinson's drawing image classifiers.")
    parser.add_argument("--data-dir", default="data/raw/parkinsons-drawings", help="Path to Kaggle dataset files.")
    parser.add_argument(
        "--model",
        default="mobilenetv2",
        choices=["baseline_cnn", "mobilenetv2"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--drawing-type",
        default="all",
        choices=["all", "spiral", "wave"],
        help="Train on spiral, wave, or all drawing images.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        drawing_type=args.drawing_type,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = create_model(args.model, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_accuracy = -1.0
    best_state = None
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate_epoch(model, val_loader, criterion, device) if val_loader else None
        selected_accuracy = val_result.accuracy if val_result else train_result.accuracy

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "val_loss": val_result.loss if val_result else None,
                "val_accuracy": val_result.accuracy if val_result else None,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_result.loss:.4f} train_acc={train_result.accuracy:.4f} "
            + (
                f"val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f}"
                if val_result
                else "val_loss=n/a val_acc=n/a"
            )
        )

        if selected_accuracy > best_val_accuracy:
            best_val_accuracy = selected_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    evaluation_loader = test_loader or val_loader
    test_metrics = {}
    if evaluation_loader is not None:
        y_true, y_probability, _paths = collect_predictions(model, evaluation_loader, device)
        test_metrics = compute_binary_metrics(y_true, y_probability)

    metrics = {
        "model_name": args.model,
        "drawing_type": args.drawing_type,
        "image_size": args.image_size,
        "best_validation_accuracy": best_val_accuracy,
        "history": history,
        "test_metrics": test_metrics,
    }

    checkpoint_path = output_dir / f"{args.model}_parkinsons.pt"
    metrics_path = output_dir / f"{args.model}_metrics.json"
    save_checkpoint(
        checkpoint_path,
        model,
        model_name=args.model,
        image_size=args.image_size,
        drawing_type=args.drawing_type,
        metrics=metrics,
    )
    save_metrics(metrics_path, metrics)

    aggregate_metrics_path = output_dir / "metrics.json"
    aggregate_metrics = load_metrics(aggregate_metrics_path) or {}
    aggregate_metrics[args.model] = metrics
    save_metrics(aggregate_metrics_path, aggregate_metrics)

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Updated metrics summary: {aggregate_metrics_path}")


if __name__ == "__main__":
    main()

