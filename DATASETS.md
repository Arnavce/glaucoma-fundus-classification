# Datasets

Download the images from the providers. This repository distributes code and aggregate
metrics. Use each dataset according to its provider's terms.

| Dataset | Where to get it | Role |
| --- | --- | --- |
| ACRIMA | [Assigned Kaggle archive](https://www.kaggle.com/datasets/ayush02102001/glaucoma-classification-datasets) | 705 images, split into 493 train, 106 validation, 106 test |
| DRISHTI-GS | [Assigned Kaggle archive](https://www.kaggle.com/datasets/ayush02102001/glaucoma-classification-datasets) and [dataset homepage](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) | 101 external images |
| RIM-ONE DL r3 | [Assigned Kaggle archive](https://www.kaggle.com/datasets/ayush02102001/glaucoma-classification-datasets) and [RIM-ONE DL repository](https://github.com/miag-ull/rim-one-dl) | 137 processed external images with an `r3_` prefix |
| ORIGA | [Assigned Kaggle archive](https://www.kaggle.com/datasets/ayush02102001/glaucoma-classification-datasets) | 650 external images |
| REFUGE | [REFUGE Challenge](https://refuge.grand-challenge.org/Details/) | 400 filename-labeled images, split into 280 train, 60 validation, 60 test |

The REFUGE experiment used the supplied ZIP with the archive path
`REFUGE2/train/images`, containing 40 `gNNNN.jpg` glaucoma images and 360
`nNNNN.jpg` normal images. Only those 400 images were used. The challenge page
describes the source benchmark. The report's test split is a new split of the labeled
subset rather than the published challenge test set.

Run the dataset preparation utility with either or both locally downloaded ZIP files.

```sh
python glaucoma/prepare_data.py --combined /path/to/combined-glaucoma.zip
python glaucoma/prepare_data.py --refuge /path/to/refuge-archive.zip
```

The script extracts just the selected image trees into the ignored `data/glaucoma`
directory. The archive typically has separate copies of partitioned and
unpartitioned ACRIMA and RIM-ONE data. We use the unpartitioned copy of each.
