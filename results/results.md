| Pipeline | Steps | Train acc. | Test acc. | Test F1 (macro) | Nested CV acc. (mean ± std) |
|---|---|---|---|---|---|
| Pipeline 1 | K-Means → Otsu → GLCM | 76.4% | 76.1% | 0.723 | 73.1% ± 5.1% |
| Pipeline 2 | K-Means → GLCM | 83.0% | 71.7% | 0.693 | 68.8% ± 5.8% |
| Pipeline 3 | Otsu → HOG | 96.7% | 78.3% | 0.753 | 80.5% ± 4.8% |
