"""Run all three pipelines and write metrics and figures to ``results/``.

Usage:
    python run_experiments.py [--data-dir data/brain_tumor_dataset] [--cv-repeats 10]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brain_tumor.classification import holdout_evaluation, repeated_cv_evaluation
from brain_tumor.data import CLASS_NAMES, load_dataset
from brain_tumor.pipelines import PIPELINES
from brain_tumor.preprocessing import preprocess
from brain_tumor.visualization import plot_confusion_matrices, plot_pipeline_stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/brain_tumor_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cv-repeats", type=int, default=10, help="repeats of 5-fold CV (0 to skip)")
    parser.add_argument("--keep-duplicates", action="store_true", help="do not remove duplicate images")
    return parser.parse_args()


def results_table(results: dict) -> str:
    lines = [
        "| Pipeline | Steps | Train acc. | Test acc. | Test F1 (macro) | Nested CV acc. (mean ± std) |",
        "|---|---|---|---|---|---|",
    ]
    for pipeline in PIPELINES:
        r = results[pipeline.key]
        holdout, cv = r["holdout"], r.get("cross_validation")
        cv_text = f"{cv['accuracy']['mean']:.1%} ± {cv['accuracy']['std']:.1%}" if cv else "–"
        lines.append(
            f"| {pipeline.name} | {pipeline.steps} | {holdout['train_accuracy']:.1%} "
            f"| {holdout['test_accuracy']:.1%} | {holdout['report']['macro avg']['f1-score']:.3f} | {cv_text} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data_dir, deduplicate=not args.keep_duplicates)
    print(f"Loaded {len(dataset)} images {dataset.class_counts()}; removed {len(dataset.duplicates)} duplicates")

    start = time.perf_counter()
    images = [preprocess(image) for image in dataset.images]
    print(f"Pre-processing: {time.perf_counter() - start:.1f} s")

    results = {}
    for pipeline in PIPELINES:
        start = time.perf_counter()
        X = pipeline.transform(images)
        extraction_time = time.perf_counter() - start

        holdout = holdout_evaluation(X, dataset.labels)
        cv = repeated_cv_evaluation(X, dataset.labels, n_repeats=args.cv_repeats) if args.cv_repeats else None
        results[pipeline.key] = {
            "name": pipeline.name,
            "steps": pipeline.steps,
            "n_features": int(X.shape[1]),
            "feature_extraction_seconds": round(extraction_time, 2),
            "holdout": holdout,
            "cross_validation": cv,
        }

        print(f"\n=== {pipeline.name}: {pipeline.steps} ({X.shape[1]} features, {extraction_time:.1f} s) ===")
        print(f"Best parameters: {holdout['best_params']}")
        print(f"Train accuracy: {holdout['train_accuracy']:.2%} | Test accuracy: {holdout['test_accuracy']:.2%}")
        print(holdout["report_text"])
        if cv:
            print(f"Nested CV ({cv['n_folds']} folds): accuracy {cv['accuracy']['mean']:.2%} ± {cv['accuracy']['std']:.2%}")

    for r in results.values():
        r["holdout"].pop("report_text")
    summary = {
        "n_images": len(dataset),
        "class_counts": dataset.class_counts(),
        "duplicates_removed": len(dataset.duplicates),
        "pipelines": results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "results.md").write_text(results_table(results) + "\n", encoding="utf-8")

    plot_confusion_matrices(
        {p.name: np.array(results[p.key]["holdout"]["confusion_matrix"]) for p in PIPELINES},
        CLASS_NAMES,
    ).savefig(figures_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")

    example = next(i for i, label in enumerate(dataset.labels) if label == 1)
    plot_pipeline_stages(dataset.images[example]).savefig(
        figures_dir / "pipeline_stages.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")

    print("\n" + results_table(results))
    print(f"\nResults written to {args.output_dir}/")


if __name__ == "__main__":
    main()
