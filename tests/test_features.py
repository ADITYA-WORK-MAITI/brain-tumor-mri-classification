import cv2
import numpy as np

from brain_tumor.features import GLCM_OFFSETS, glcm, glcm_features, hog_features, image_gradients


def reference_glcm(image, offset, levels=256):
    """Direct nested-loop GLCM, used to check the vectorised version."""
    dr, dc = offset
    rows, cols = image.shape
    matrix = np.zeros((levels, levels))
    for r in range(rows):
        for c in range(cols):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                matrix[image[r, c], image[nr, nc]] += 1
    return matrix / matrix.sum()


def test_glcm_matches_reference_loop():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(20, 17), dtype=np.uint8)
    for offset in GLCM_OFFSETS:
        np.testing.assert_allclose(glcm(image, offset), reference_glcm(image, offset))


def test_glcm_features_on_known_image():
    # Vertical stripes 0, 255, 0, 255, ...: every horizontal pair differs by 255,
    # every vertical pair is identical.
    image = np.tile(np.array([0, 255], dtype=np.uint8), (16, 8))
    mean, std, contrast, entropy, energy, homogeneity, correlation = glcm_features(image)
    assert mean == 127.5 and std == 127.5
    # 0 deg and both diagonals: contrast 255^2; 90 deg: 0.
    np.testing.assert_allclose(contrast, 3 * 255**2 / 4)
    np.testing.assert_allclose(homogeneity, (3 / 256 + 1) / 4)
    np.testing.assert_allclose(correlation, (-1 - 1 - 1 + 1) / 4, atol=1e-2)
    assert 0 < energy <= 1 and entropy > 0


def test_gradients_match_opencv_sobel_ksize1():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=(32, 24), dtype=np.uint8)
    gx, gy = image_gradients(image)
    np.testing.assert_allclose(gx, cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1))
    np.testing.assert_allclose(gy, cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1))


def test_hog_shape_and_normalisation():
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    features = hog_features(image)
    assert features.shape == (15 * 15 * 36,)
    block_norms = np.linalg.norm(features.reshape(-1, 36), axis=1)
    np.testing.assert_allclose(block_norms, 1.0, atol=1e-6)


def test_hog_vertical_edge_votes_into_zero_degree_bin():
    image = np.zeros((16, 16), dtype=np.uint8)
    image[:, 8:] = 255  # vertical edge -> horizontal gradient -> orientation 0
    features = hog_features(image, cell_size=8, block_size=1)
    histograms = features.reshape(-1, 9)
    edge_cells = histograms[histograms.sum(axis=1) > 0]
    assert np.all(np.argmax(edge_cells, axis=1) == 0)
