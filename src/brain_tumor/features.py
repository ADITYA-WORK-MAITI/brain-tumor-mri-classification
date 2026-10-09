"""Feature extraction: GLCM texture statistics and HOG descriptors."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

GLCM_LEVELS = 256
# (row, column) offsets for 0, 45, 90 and 135 degrees at distance 1.
GLCM_OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
GLCM_FEATURE_NAMES = (
    "mean",
    "std",
    "contrast",
    "entropy",
    "energy",
    "homogeneity",
    "correlation",
)


def glcm(image: np.ndarray, offset: tuple[int, int], levels: int = GLCM_LEVELS) -> np.ndarray:
    """Normalised gray-level co-occurrence matrix for one pixel offset.

    Entry ``[i, j]`` is the probability that a pixel with intensity ``i`` has a
    neighbour with intensity ``j`` at the given (row, column) offset.
    """
    dr, dc = offset
    rows, cols = image.shape
    r0, r1 = max(0, -dr), rows - max(0, dr)
    c0, c1 = max(0, -dc), cols - max(0, dc)
    reference = image[r0:r1, c0:c1].astype(np.int64)
    neighbour = image[r0 + dr : r1 + dr, c0 + dc : c1 + dc].astype(np.int64)

    counts = np.bincount((reference * levels + neighbour).ravel(), minlength=levels * levels)
    matrix = counts.reshape(levels, levels).astype(np.float64)
    return matrix / matrix.sum()


def glcm_features(image: np.ndarray, distance: int = 1) -> np.ndarray:
    """Seven texture features: intensity mean and standard deviation, plus
    GLCM contrast, entropy, energy, homogeneity and correlation averaged over
    the four directions (see ``GLCM_FEATURE_NAMES``).

    Energy is the angular second moment, sum(p^2).
    """
    i, j = np.indices((GLCM_LEVELS, GLCM_LEVELS))
    levels = np.arange(GLCM_LEVELS)
    per_direction = []

    for dr, dc in GLCM_OFFSETS:
        p = glcm(image, (dr * distance, dc * distance))
        p_i, p_j = p.sum(axis=1), p.sum(axis=0)
        mu_i, mu_j = np.sum(levels * p_i), np.sum(levels * p_j)
        sigma_i = np.sqrt(np.sum((levels - mu_i) ** 2 * p_i))
        sigma_j = np.sqrt(np.sum((levels - mu_j) ** 2 * p_j))

        nonzero = p[p > 0]
        contrast = np.sum((i - j) ** 2 * p)
        entropy = -np.sum(nonzero * np.log2(nonzero))
        energy = np.sum(p**2)
        homogeneity = np.sum(p / (1.0 + np.abs(i - j)))
        if sigma_i > 0 and sigma_j > 0:
            correlation = np.sum((i - mu_i) * (j - mu_j) * p) / (sigma_i * sigma_j)
        else:  # undefined for a constant image
            correlation = 0.0
        per_direction.append((contrast, entropy, energy, homogeneity, correlation))

    texture = np.mean(per_direction, axis=0)
    return np.concatenate(([image.mean(), image.std()], texture))


def image_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical gradients with the centred [-1, 0, 1] kernel.

    Borders are reflected without repeating the edge pixel.
    """
    padded = np.pad(image.astype(np.float64), 1, mode="reflect")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    return gx, gy


def hog_features(
    image: np.ndarray, cell_size: int = 8, block_size: int = 2, n_bins: int = 9, eps: float = 1e-6
) -> np.ndarray:
    """Histogram of Oriented Gradients descriptor.

    1. Gradient magnitude and unsigned orientation in [0, 180) for each pixel.
    2. Per ``cell_size`` x ``cell_size`` cell, a histogram of orientations
       weighted by magnitude. Each vote is split linearly between the two
       nearest bins.
    3. Overlapping ``block_size`` x ``block_size`` blocks of cells are
       L2-normalised for robustness to contrast changes and concatenated.

    A 128x128 image with the defaults gives 15 x 15 blocks x 36 values = 8100 features.
    """
    gx, gy = image_gradients(image)
    magnitude = np.hypot(gx, gy)
    orientation = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    n_cells_y, n_cells_x = image.shape[0] // cell_size, image.shape[1] // cell_size
    height, width = n_cells_y * cell_size, n_cells_x * cell_size
    magnitude, orientation = magnitude[:height, :width], orientation[:height, :width]

    position = orientation / (180.0 / n_bins)
    lower_bin = np.floor(position).astype(np.int64) % n_bins
    upper_bin = (lower_bin + 1) % n_bins
    upper_weight = position - np.floor(position)

    cell_index = (np.arange(height) // cell_size)[:, None] * n_cells_x + (np.arange(width) // cell_size)[None, :]
    n_entries = n_cells_y * n_cells_x * n_bins
    histograms = np.bincount(
        (cell_index * n_bins + lower_bin).ravel(), (magnitude * (1 - upper_weight)).ravel(), n_entries
    ) + np.bincount((cell_index * n_bins + upper_bin).ravel(), (magnitude * upper_weight).ravel(), n_entries)
    histograms = histograms.reshape(n_cells_y, n_cells_x, n_bins)

    blocks = sliding_window_view(histograms, (block_size, block_size, n_bins))[:, :, 0]
    blocks = blocks.reshape(-1, block_size * block_size * n_bins)
    blocks = blocks / (np.linalg.norm(blocks, axis=1, keepdims=True) + eps)
    return blocks.ravel()
