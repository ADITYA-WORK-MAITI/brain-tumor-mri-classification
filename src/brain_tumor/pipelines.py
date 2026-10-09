"""The three feature-extraction pipelines compared in this project.

All three share the same pre-processing (``preprocessing.preprocess``) and the
same SVM classifier. They differ in segmentation and feature extraction:

* Pipeline 1 (the original paper's method): K-Means -> Otsu -> GLCM
* Pipeline 2: K-Means -> GLCM (GLCM sees 4 gray levels instead of a binary mask)
* Pipeline 3: Otsu -> HOG (shape and edge features instead of texture)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .features import glcm_features, hog_features
from .segmentation import otsu_binarize, quantize_kmeans

KMEANS_CLUSTERS = 4  # intended to separate background, CSF, gray/white matter and tumor


def pipeline_1_features(image: np.ndarray) -> np.ndarray:
    """K-Means (k=4) -> Otsu thresholding of the clustered image -> GLCM."""
    clustered = quantize_kmeans(image, KMEANS_CLUSTERS)
    return glcm_features(otsu_binarize(clustered))


def pipeline_2_features(image: np.ndarray) -> np.ndarray:
    """K-Means (k=4) -> GLCM on the 4-level clustered image."""
    return glcm_features(quantize_kmeans(image, KMEANS_CLUSTERS))


def pipeline_3_features(image: np.ndarray) -> np.ndarray:
    """Otsu thresholding of the pre-processed image -> HOG."""
    return hog_features(otsu_binarize(image))


@dataclass(frozen=True)
class FeaturePipeline:
    key: str
    name: str
    steps: str
    extract: Callable[[np.ndarray], np.ndarray]

    def transform(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.extract(image) for image in images])


PIPELINES = (
    FeaturePipeline("pipeline_1", "Pipeline 1", "K-Means → Otsu → GLCM", pipeline_1_features),
    FeaturePipeline("pipeline_2", "Pipeline 2", "K-Means → GLCM", pipeline_2_features),
    FeaturePipeline("pipeline_3", "Pipeline 3", "Otsu → HOG", pipeline_3_features),
)
