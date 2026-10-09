"""Figures for the README and the notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .features import image_gradients
from .preprocessing import equalize_histogram, median_blur_3x3, resize, to_grayscale
from .segmentation import otsu_binarize, quantize_kmeans


def plot_pipeline_stages(image_bgr: np.ndarray) -> plt.Figure:
    """Show every intermediate image of the three pipelines for one MRI scan."""
    gray = to_grayscale(image_bgr)
    denoised = median_blur_3x3(gray)
    equalized = equalize_histogram(denoised)
    resized = resize(equalized)
    clustered = quantize_kmeans(resized)
    gx, gy = image_gradients(otsu_binarize(resized))

    stages = [
        ("Original", image_bgr[..., ::-1], None),
        ("Grayscale", gray, "gray"),
        ("Median blur 3×3", denoised, "gray"),
        ("Histogram equalization", equalized, "gray"),
        ("Resized 128×128", resized, "gray"),
        ("K-Means, k=4\n(Pipelines 1, 2)", clustered, "gray"),
        ("Otsu on K-Means\n(Pipeline 1)", otsu_binarize(clustered), "gray"),
        ("Otsu on image\n(Pipeline 3)", otsu_binarize(resized), "gray"),
        ("Gradient magnitude\n(HOG input, Pipeline 3)", np.hypot(gx, gy), "magma"),
    ]
    fig, axes = plt.subplots(1, len(stages), figsize=(2.2 * len(stages), 2.8))
    for ax, (title, img, cmap) in zip(axes, stages):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_confusion_matrices(matrices: dict[str, np.ndarray], class_names: tuple[str, ...]) -> plt.Figure:
    """Side-by-side confusion matrices (rows: true class, columns: predicted class)."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 3.6))
    for ax, (title, matrix) in zip(np.atleast_1d(axes), matrices.items()):
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (r, c), value in np.ndenumerate(matrix):
            color = "white" if value > matrix.max() / 2 else "black"
            ax.text(c, r, str(value), ha="center", va="center", fontsize=13, color=color)
        accuracy = np.trace(matrix) / matrix.sum()
        ax.set_title(f"{title} (accuracy {accuracy:.1%})", fontsize=10)
        ax.set_xticks(range(len(class_names)), class_names)
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    return fig
