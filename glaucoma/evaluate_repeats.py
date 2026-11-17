"""Evaluate the three ACRIMA training seeds on each external dataset."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from train import GlaucomaCNN, evaluate, load_data

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "glaucoma"
RUNS = ROOT / "repeats"
SEEDS = (42, 43, 44)


def main():
    torch.set_num_threads(4)
    datasets = {
        "DRISHTI-GS": ("DRISHTI-GS/DRISHTI-GS", "*"),
        "RIM-ONE DL r3": ("RIM-ONE/RIM-ONE/NOT PARTITIONED", "r3_*"),
        "ORIGA": ("ORIGA/ORIGA", "*"),
    }
    loaders = {}
    for name, (path, pattern) in datasets.items():
        images, labels, _ = load_data(DATA / path, pattern=pattern)
        x = torch.from_numpy(images).permute(0, 3, 1, 2)
        loaders[name] = DataLoader(TensorDataset(x, torch.from_numpy(labels)), batch_size=32)

    results = {"standard": {}, "small": {}}
    reference_test = None
    for kind in results:
        for seed in SEEDS:
            folder = RUNS / kind / f"seed{seed}"
            metrics = json.loads((folder / "metrics.json").read_text())
            with (folder / "test_predictions.csv").open() as stream:
                test_rows = [(row["image"], row["label"]) for row in csv.DictReader(stream)]
            if reference_test is None:
                reference_test = test_rows
            elif test_rows != reference_test:
                raise ValueError(f"Inconsistent ACRIMA test split for {kind}, seed {seed}")

            model = GlaucomaCNN(attention=True, separable=kind == "small")
            model.load_state_dict(torch.load(folder / "glaucoma_cnn.pt", weights_only=True,
                                             map_location="cpu"))
            run = {"acrima_accuracy": metrics["test_accuracy"], "acrima_auroc": metrics["test_auc"],
                   "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
                   "confusion_matrix": metrics["confusion_matrix"], "parameters": metrics["parameters"],
                   "external": {}}
            for name, loader in loaders.items():
                labels, scores = evaluate(model, loader, "cpu")
                run["external"][name] = {
                    "auroc": float(roc_auc_score(labels, scores)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= 0.5)),
                }
            results[kind][seed] = run
            print(f"Evaluated {kind} seed {seed}", flush=True)
    (RUNS / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
