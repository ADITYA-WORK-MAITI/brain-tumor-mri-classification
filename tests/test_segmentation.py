import cv2
import numpy as np

from brain_tumor.segmentation import kmeans_intensity, otsu_binarize, otsu_threshold, quantize_kmeans


def test_otsu_matches_opencv():
    rng = np.random.default_rng(1)
    for _ in range(20):
        # Bimodal images with random class means.
        low, high = sorted(rng.integers(20, 235, size=2))
        image = np.concatenate([rng.normal(low, 12, 2000), rng.normal(high, 12, 3000)])
        image = np.clip(image, 0, 255).astype(np.uint8).reshape(50, 100)
        _, expected = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        np.testing.assert_array_equal(otsu_binarize(image), expected)


def test_otsu_on_constant_image():
    assert otsu_threshold(np.full((5, 5), 9, dtype=np.uint8)) == 0


def test_kmeans_recovers_four_levels():
    rng = np.random.default_rng(2)
    levels = np.array([0, 70, 140, 220])
    truth = rng.integers(0, 4, size=(64, 64))
    image = np.clip(levels[truth] + rng.normal(0, 5, truth.shape), 0, 255).astype(np.uint8)

    labels, centroids = kmeans_intensity(image, k=4)
    np.testing.assert_allclose(centroids, levels, atol=3)
    assert np.mean(labels == truth) > 0.99  # labels are sorted by intensity


def test_kmeans_handles_dominant_background():
    # 90 % black background used to produce duplicate initial centroids.
    image = np.zeros((100, 100), dtype=np.uint8)
    image[40:60, 40:60] = 120
    image[45:55, 45:55] = 250
    quantized = quantize_kmeans(image, k=4)
    assert set(np.unique(quantized)) >= {0, 120, 250}
