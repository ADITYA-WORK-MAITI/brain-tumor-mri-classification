"""Classical image-processing pipelines for brain tumor detection in MRI scans.

Every image-processing step (grayscale conversion, median filtering, histogram
equalization, K-Means, Otsu thresholding, GLCM and HOG) is implemented from
scratch with NumPy. OpenCV is used only for image I/O and resizing, and
scikit-learn only for the SVM classifier and evaluation utilities.
"""

__version__ = "1.0.0"
