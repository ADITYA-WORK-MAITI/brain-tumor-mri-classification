# Dataset

This project uses the **Brain MRI Images for Brain Tumor Detection** dataset
(N. Chakrabarty, Kaggle):
<https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection>

The images are not included in this repository. Download `archive.zip` from
Kaggle and extract it here. The code expects this layout:

```
data/
└── brain_tumor_dataset/
    ├── no/    98 MRI scans without a tumor
    └── yes/   155 MRI scans with a tumor
```

The archive also contains a second copy of the same `yes/` and `no/` folders
at its top level. Those copies are not used.

To use a different location, run `python run_experiments.py --data-dir <path>`.

## Duplicate images

25 of the 253 files duplicate another image in the same class (22 groups of
identical images). `brain_tumor.data.load_dataset` keeps one copy of each, so
all experiments use **228 unique images (87 no tumor, 141 tumor)**. Without
this step, a copy of a training image can land in the test set and make the
test accuracy look better than it is.
