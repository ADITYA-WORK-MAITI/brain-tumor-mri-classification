"""Pre-processing: grayscale conversion, denoising, contrast enhancement, resizing."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

IMAGE_SIZE = (128, 128)


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Luminance-weighted grayscale conversion (ITU-R BT.601 weights)."""
    b, g, r = (image_bgr[..., c].astype(np.float64) for c in range(3))
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def median_blur_3x3(image: np.ndarray) -> np.ndarray:
    """3x3 median filter with replicated borders.

    Median filtering removes salt-and-pepper noise while preserving edges,
    which matters for tumor boundaries.
    """
    padded = np.pad(image, 1, mode="edge")
    windows = sliding_window_view(padded, (3, 3))
    return np.median(windows, axis=(-2, -1)).astype(image.dtype)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization of an 8-bit image.

    Each intensity v is mapped through the cumulative distribution function:
    ``round((cdf(v) - cdf_min) / (N - cdf_min) * 255)``, where ``cdf_min`` is
    the CDF value of the darkest intensity present and ``N`` the pixel count.
    """
    hist = np.bincount(image.ravel(), minlength=256)
    cdf = np.cumsum(hist)
    cdf_min = cdf[np.flatnonzero(hist)[0]]
    if cdf_min == image.size:  # constant image: nothing to equalize
        return image.copy()
    lut = np.rint((cdf - cdf_min) * 255.0 / (image.size - cdf_min))
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[image]


def resize(image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize with area interpolation, which avoids aliasing when shrinking."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess(image_bgr: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Full pre-processing chain: grayscale -> median blur -> equalization -> resize."""
    gray = to_grayscale(image_bgr)
    denoised = median_blur_3x3(gray)
    equalized = equalize_histogram(denoised)
    return resize(equalized, size)
