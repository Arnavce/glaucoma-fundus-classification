"""Test both ACRIMA checkpoints on each external dataset separately."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from train import GlaucomaCNN, evaluate, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/glaucoma"))
    parser.add_argument("--out", type=Path, default=Path("glaucoma/results_attention/external_metrics.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)

    models = {}
    for name, attention, directory in (("Baseline", False, "results"),
                                       ("Channel attention", True, "results_attention")):
        checkpoint = Path(__file__).parent / directory / "glaucoma_cnn.pt"
        model = GlaucomaCNN(attention=attention)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        models[name] = model

    datasets = (
        ("DRISHTI-GS", "DRISHTI-GS/DRISHTI-GS", "*"),
        ("RIM-ONE DL r3 subset", "RIM-ONE/RIM-ONE/NOT PARTITIONED", "r3_*"),
        ("ORIGA", "ORIGA/ORIGA", "*"),
    )
    results = {}
    for name, location, pattern in datasets:
        images, labels, _ = load_data(args.data / location, pattern=pattern)
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        loader = DataLoader(TensorDataset(images, torch.from_numpy(labels)), batch_size=32)
        results[name] = {"class_counts_normal_glaucoma": np.bincount(labels, minlength=2).tolist()}
        for model_name, model in models.items():
            y, scores = evaluate(model, loader, "cpu")
            predicted = (scores >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
            results[name][model_name] = {
                "auroc": float(roc_auc_score(y, scores)),
                "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
                "sensitivity": float(tp / (tp + fn)),
                "specificity": float(tn / (tn + fp)),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
