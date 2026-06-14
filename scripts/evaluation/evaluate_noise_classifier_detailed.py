from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.inference.classifier_inference import ClassifierInference
from src.noise_classifier.dataset import LABELS, NoiseDataset
from src.noise_classifier.transforms import get_val_transforms


DEFAULT_CHECKPOINT = Path("models/classifiers/noise_classifier_best.pt")
DEFAULT_DATA_DIR = Path("data/synthetic")
DEFAULT_OUTPUT_DIR = Path("outputs/classifier_misclassified")


@dataclass(frozen=True)
class MisclassifiedSample:
    image_path: Path
    true_label: str
    predicted_label: str
    confidence: float
    topk: list[tuple[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed evaluation for the noise classifier.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--save-limit-per-group", type=int, default=10)
    return parser.parse_args()


def create_loader(data_dir: Path, labels: list[str], batch_size: int, num_workers: int) -> tuple[NoiseDataset, DataLoader]:
    dataset = NoiseDataset(root_dir=data_dir, labels=labels, transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataset, loader


def format_topk(topk: list[tuple[str, float]]) -> str:
    return ", ".join(f"{label}={score:.4f}" for label, score in topk)


def evaluate_classifier(
    inference: ClassifierInference,
    dataset: NoiseDataset,
    loader: DataLoader,
    labels: list[str],
    top_k: int,
) -> tuple[np.ndarray, list[MisclassifiedSample]]:
    if inference.model is None:
        inference.load_model()

    assert inference.model is not None
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    misclassified: list[MisclassifiedSample] = []
    sample_offset = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating classifier", unit="batch"):
            batch_size = images.size(0)
            images = images.to(inference.device)
            targets = targets.to(inference.device)

            logits = inference.model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            topk_count = min(top_k, probabilities.shape[1])
            topk_scores, topk_indices = probabilities.topk(topk_count, dim=1)

            targets_np = targets.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            confidences_np = confidences.cpu().numpy()
            topk_scores_np = topk_scores.cpu().numpy()
            topk_indices_np = topk_indices.cpu().numpy()

            for batch_index in range(batch_size):
                true_index = int(targets_np[batch_index])
                pred_index = int(predictions_np[batch_index])
                confusion[true_index, pred_index] += 1

                if true_index != pred_index:
                    dataset_index = sample_offset + batch_index
                    image_path, _ = dataset.samples[dataset_index]
                    topk = [
                        (labels[int(class_index)], float(score))
                        for class_index, score in zip(topk_indices_np[batch_index], topk_scores_np[batch_index])
                    ]
                    misclassified.append(
                        MisclassifiedSample(
                            image_path=image_path,
                            true_label=labels[true_index],
                            predicted_label=labels[pred_index],
                            confidence=float(confidences_np[batch_index]),
                            topk=topk,
                        )
                    )

            sample_offset += batch_size

    return confusion, misclassified


def save_misclassified_samples(
    samples: list[MisclassifiedSample],
    output_dir: Path,
    save_limit_per_group: int,
) -> None:
    grouped_counts: dict[str, int] = {}

    for sample in samples:
        group_name = f"true_{sample.true_label}_pred_{sample.predicted_label}"
        count = grouped_counts.get(group_name, 0)
        if count >= save_limit_per_group:
            continue

        grouped_counts[group_name] = count + 1
        sample_dir = output_dir / group_name / f"{count + 1:02d}_{sample.image_path.stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(sample.image_path) as image:
            image.convert("RGB").save(sample_dir / sample.image_path.name)

        metadata_lines = [
            f"true_label: {sample.true_label}",
            f"predicted_label: {sample.predicted_label}",
            f"confidence: {sample.confidence:.6f}",
            f"topk: {format_topk(sample.topk)}",
            f"source_path: {sample.image_path.resolve()}",
        ]
        (sample_dir / "prediction.txt").write_text("\n".join(metadata_lines) + "\n")


def print_summary(
    confusion: np.ndarray,
    labels: list[str],
    misclassified: list[MisclassifiedSample],
) -> None:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    overall_accuracy = correct / max(1, total)

    print()
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print("Per-class accuracy:")
    for idx, label in enumerate(labels):
        row_total = int(confusion[idx].sum())
        class_acc = confusion[idx, idx] / row_total if row_total > 0 else 0.0
        print(f"- {label}: {class_acc:.4f}")

    print("Confusion matrix:")
    print("labels:", labels)
    print(confusion)

    speckle_index = labels.index("speckle") if "speckle" in labels else None
    if speckle_index is not None:
        row = confusion[speckle_index]
        row_total = int(row.sum())
        print("Speckle misclassification breakdown:")
        for idx, label in enumerate(labels):
            if idx == speckle_index:
                continue
            count = int(row[idx])
            ratio = count / row_total if row_total > 0 else 0.0
            print(f"- speckle -> {label}: {count} ({ratio:.4f})")

    print("Misclassified samples:")
    if not misclassified:
        print("- none")
        return

    for sample in misclassified:
        print(
            f"- {sample.image_path.name} | true={sample.true_label} pred={sample.predicted_label} "
            f"conf={sample.confidence:.4f} topk=[{format_topk(sample.topk)}]"
        )


def main() -> None:
    args = parse_args()
    inference = ClassifierInference(model_path=args.checkpoint, device=args.device)
    inference.load_model()

    labels = list(LABELS)
    dataset, loader = create_loader(
        data_dir=args.data_dir.expanduser().resolve(),
        labels=labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    confusion, misclassified = evaluate_classifier(
        inference=inference,
        dataset=dataset,
        loader=loader,
        labels=labels,
        top_k=args.top_k,
    )

    save_misclassified_samples(
        samples=misclassified,
        output_dir=args.output_dir.expanduser().resolve(),
        save_limit_per_group=args.save_limit_per_group,
    )
    print_summary(confusion=confusion, labels=labels, misclassified=misclassified)


if __name__ == "__main__":
    main()
