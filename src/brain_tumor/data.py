"""Loading the "Brain MRI Images for Brain Tumor Detection" dataset."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Folder name -> class label. 1 = tumor, 0 = no tumor.
CLASS_FOLDERS = {"no": 0, "yes": 1}
CLASS_NAMES = ("no tumor", "tumor")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class MRIDataset:
    images: list[np.ndarray]  # BGR uint8 images at their original resolution
    labels: np.ndarray  # shape (n,), values in {0, 1}
    paths: list[Path]
    duplicates: list[tuple[Path, Path]] = field(default_factory=list)  # (dropped, kept)

    def __len__(self) -> int:
        return len(self.images)

    def class_counts(self) -> dict[str, int]:
        return {name: int(np.sum(self.labels == label)) for name, label in zip(CLASS_NAMES, (0, 1))}


def load_dataset(data_dir: str | Path, deduplicate: bool = True) -> MRIDataset:
    """Load all images from ``data_dir/no`` and ``data_dir/yes``.

    Files are read in sorted order so that results are reproducible across
    operating systems. When ``deduplicate`` is True, images with identical
    pixel content are kept only once; otherwise a copy of a training image can
    end up in the test set and inflate the reported accuracy.
    """
    data_dir = Path(data_dir)
    images, labels, paths = [], [], []
    duplicates = []
    seen: dict[str, tuple[Path, int]] = {}

    for folder, label in CLASS_FOLDERS.items():
        class_dir = data_dir / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}. See data/README.md for the expected layout."
            )
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                warnings.warn(f"Skipping unreadable image: {path}")
                continue

            if deduplicate:
                digest = hashlib.sha1(image.tobytes() + str(image.shape).encode()).hexdigest()
                if digest in seen:
                    kept_path, kept_label = seen[digest]
                    if kept_label != label:
                        raise ValueError(f"Identical images with different labels: {kept_path}, {path}")
                    duplicates.append((path, kept_path))
                    continue
                seen[digest] = (path, label)

            images.append(image)
            labels.append(label)
            paths.append(path)

    if not images:
        raise ValueError(f"No images found in {data_dir}")
    return MRIDataset(images, np.asarray(labels, dtype=np.int64), paths, duplicates)
