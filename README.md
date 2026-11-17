# Glaucoma Fundus Classification

A Group 6 machine learning project comparing three CNN architectures for binary
classification of retinal fundus images. Grad-CAM visualizes regions that
contribute to a prediction. This repository contains the classification
component of the group project.

| Model | Training flag | Prediction flag | Trainable parameters |
| --- | --- | --- | ---: |
| Baseline CNN | omit both flags | `baseline` | 23,826 |
| Standard SE CNN | `--attention` | `standard-se` | 24,922 |
| Small SE CNN | `--small-se` | `small-se` | 4,517 |

The SE (Squeeze-and-Excitation) module learns channel weights for each image.
Small SE preserves the attention module and replaces each standard convolution
with a depthwise 3 × 3 filter and a pointwise 1 × 1 channel mixer.

## Setup and datasets

Use Python 3.12. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

[Dataset sources and setup](DATASETS.md) covers ACRIMA, DRISHTI-GS, RIM-ONE DL r3,
ORIGA, and REFUGE. Raw images, archives, model weights, and credentials are
excluded from version control. Training produces checkpoints locally.

## Train and predict

```sh
python glaucoma/train.py --data 'data/glaucoma/ACRIMA/ACRIMA/NOT PARTITIONED' \
  --small-se --epochs 24 --seed 42 --device cpu
python glaucoma/predict.py --architecture small-se \
  --checkpoint glaucoma/results_small_se/glaucoma_cnn.pt \
  --image /path/to/fundus.jpg --gradcam /tmp/overlay.png
```

Select the training and prediction flags from the model table to switch architectures.
The default seed-42 output folders are `glaucoma/results`, `glaucoma/results_attention`,
and `glaucoma/results_small_se`, respectively. Pass `--out` to set an explicit run
folder. To train on REFUGE, set `--data data/glaucoma/refuge_kaggle_labeled` and use
a distinct output folder for each architecture and seed. The split itself stays
fixed while `--seed` changes training randomness.

To reproduce the paired external evaluations, save the ACRIMA Standard SE and
Small SE checkpoints for seeds 42, 43, and 44 under
`glaucoma/repeats/standard/seedNN` and `glaucoma/repeats/small/seedNN`, then run:

```sh
python glaucoma/evaluate_repeats.py
```

## Results and limits

Small SE averaged 93.7% accuracy on held-out ACRIMA images, compared with 91.5%
for Standard SE across three matched runs, using 81.9% fewer trainable parameters.
Both variants struggled on the three external datasets. On the labeled REFUGE
subset, Small SE had higher mean AUROC and sensitivity in the matched runs,
while Standard SE had higher overall accuracy. The REFUGE test set had only six
glaucoma cases, and some runs predicted a single class for all images.

[ACRIMA and external results](metrics/summary.json) ·
[Baseline metrics](metrics/acrima_baseline.json) ·
[Matched REFUGE runs](metrics/refuge_all_architectures.csv) ·
[All Small SE REFUGE runs](metrics/refuge_small_se_all_seeds.csv)
