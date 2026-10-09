"""Segmentation: intensity K-Means clustering and Otsu thresholding."""

from __future__ import annotations

import numpy as np


def kmeans_intensity(
    image: np.ndarray, k: int = 4, max_iter: int = 100, tol: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster pixel intensities into ``k`` groups with Lloyd's algorithm.

    Centroids are initialised evenly across the image's intensity range, which
    is deterministic and never produces two identical starting centroids. The
    returned clusters are sorted by intensity, so label 0 is always the darkest
    cluster (the background) and labels are comparable across images.

    Returns:
        labels: integer label map with the same shape as ``image``.
        centroids: the ``k`` cluster intensities in ascending order.
    """
    pixels = image.ravel().astype(np.float64)
    low, high = pixels.min(), pixels.max()
    centroids = low + (high - low) * (np.arange(k) + 0.5) / k

    for _ in range(max_iter):
        labels = np.argmin(np.abs(pixels[:, None] - centroids[None, :]), axis=1)
        updated = centroids.copy()
        for j in range(k):
            members = pixels[labels == j]
            if members.size:  # an empty cluster keeps its previous centroid
                updated[j] = members.mean()
        converged = np.max(np.abs(updated - centroids)) < tol
        centroids = updated
        if converged:
            break

    order = np.argsort(centroids)
    centroids = centroids[order]
    labels = np.argmin(np.abs(pixels[:, None] - centroids[None, :]), axis=1)
    return labels.reshape(image.shape), centroids


def quantize_kmeans(image: np.ndarray, k: int = 4) -> np.ndarray:
    """Replace every pixel by the intensity of its K-Means cluster centroid."""
    labels, centroids = kmeans_intensity(image, k)
    return np.rint(centroids[labels]).astype(np.uint8)


def otsu_threshold(image: np.ndarray) -> int:
    """Return the threshold t that maximises the between-class variance.

    Pixels ``<= t`` form the background class and pixels ``> t`` the
    foreground class.
    """
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    intensities = np.arange(256)

    weight_bg = np.cumsum(hist)
    weight_fg = image.size - weight_bg
    sum_bg = np.cumsum(intensities * hist)
    sum_total = sum_bg[-1]

    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():  # constant image
        return 0
    mean_bg = sum_bg[valid] / weight_bg[valid]
    mean_fg = (sum_total - sum_bg[valid]) / weight_fg[valid]

    between_variance = np.zeros(256)
    between_variance[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg - mean_fg) ** 2
    return int(np.argmax(between_variance))


def otsu_binarize(image: np.ndarray) -> np.ndarray:
    """Binarize an 8-bit image with Otsu's threshold (foreground = 255)."""
    threshold = otsu_threshold(image)
    return np.where(image > threshold, 255, 0).astype(np.uint8)
