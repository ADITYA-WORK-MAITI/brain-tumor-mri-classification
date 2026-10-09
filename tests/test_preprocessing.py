import cv2
import numpy as np
import pytest

from brain_tumor.preprocessing import equalize_histogram, median_blur_3x3, preprocess, to_grayscale


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_grayscale_matches_opencv(rng):
    image = rng.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
    expected = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert np.abs(to_grayscale(image).astype(int) - expected).max() <= 1


def test_median_blur_matches_opencv(rng):
    image = rng.integers(0, 256, size=(37, 41), dtype=np.uint8)
    np.testing.assert_array_equal(median_blur_3x3(image), cv2.medianBlur(image, 3))


def test_histogram_equalization_matches_opencv(rng):
    # A low-contrast image that does not use the full 0-255 range.
    image = rng.integers(60, 140, size=(64, 64), dtype=np.uint8)
    result = equalize_histogram(image)
    assert np.abs(result.astype(int) - cv2.equalizeHist(image)).max() <= 1
    assert result.min() == 0 and result.max() == 255


def test_equalization_of_constant_image_is_identity():
    image = np.full((8, 8), 7, dtype=np.uint8)
    np.testing.assert_array_equal(equalize_histogram(image), image)


def test_preprocess_output_shape(rng):
    image = rng.integers(0, 256, size=(300, 250, 3), dtype=np.uint8)
    out = preprocess(image)
    assert out.shape == (128, 128) and out.dtype == np.uint8
