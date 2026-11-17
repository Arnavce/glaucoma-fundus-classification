"""Train and explain a compact binary classifier for fundus photographs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ("Normal", "Glaucoma")
SIZE = 128
LABELS = {
    "normal": 0, "healthy": 0, "negative": 0, "nonglaucoma": 0,
    "glaucoma": 1, "glaucomatous": 1, "positive": 1,
}


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation gate for the final convolution block."""

    def __init__(self, channels, bottleneck):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, bottleneck), nn.ReLU(),
            nn.Linear(bottleneck, channels), nn.Sigmoid(),
        )

    def forward(self, feature_maps):
        weights = self.gate(feature_maps)
        return feature_maps * weights[:, :, None, None]


class GlaucomaCNN(nn.Module):
    def __init__(self, attention=False, separable=False):
        super().__init__()
        self.features = nn.Sequential(
            self.block(3, 16, separable), self.block(16, 32, separable),
            self.block(32, 64, separable)
        )
        self.attention = ChannelAttention(64, 8) if attention else nn.Identity()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, 2)
        )

    @staticmethod
    def block(in_channels, out_channels, separable=False):
        if separable:
            convolution = [
                nn.Conv2d(in_channels, in_channels, 3, padding=1,
                          groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
            ]
        else:
            convolution = [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)]
        return nn.Sequential(
            *convolution,
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)
        )

    def forward(self, images):
        return self.score(self.features(images))

    def score(self, feature_maps):
        return self.classifier(self.attention(feature_maps))


def preprocess(path):
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        return np.asarray(ImageOps.pad(rgb, (SIZE, SIZE), Image.Resampling.BILINEAR))


def load_data(root, manifest=None, pattern="*"):
    if manifest:
        with manifest.open(newline="", encoding="utf-8-sig") as file:
            rows = [(root / row["image"], int(row["label"])) for row in csv.DictReader(file)]
    else:
        rows = []
        for path in sorted(root.rglob(pattern)):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            directories = path.relative_to(root).parts[:-1]
            labels = [LABELS.get("".join(c for c in part.lower() if c.isalpha()))
                      for part in reversed(directories)]
            label = next((value for value in labels if value is not None), None)
            if label is not None:
                rows.append((path, label))

    seen = {}
    images, targets, paths = [], [], []
    for path, label in rows:
        if label not in (0, 1):
            raise ValueError(f"Expected label 0 or 1 for {path}")
        image = preprocess(path)
        digest = hashlib.sha256(image.tobytes()).digest()
        if digest in seen:
            if seen[digest] != label:
                raise ValueError(f"Duplicate image has conflicting class labels: {path}")
            continue
        seen[digest] = label
        images.append(image)
        targets.append(label)
        paths.append(path)
    if min(targets.count(0), targets.count(1)) < 10:
        raise ValueError("Need at least 10 unique images per class. Point --data at labeled "
                         "class folders or supply --manifest with image,label columns.")
    return np.stack(images), np.array(targets), paths


def model_input(images):
    return images.to(dtype=torch.float32).div(127.5).sub(1)


def evaluate(model, loader, device):
    model.eval()
    truth, scores = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(model_input(images.to(device)))
            scores.extend(logits.softmax(1)[:, 1].cpu().tolist())
            truth.extend(labels.tolist())
    return np.array(truth), np.array(scores)


def explain(model, image, device):
    """Weight the last spatial feature maps by mean class-score gradient."""
    model.eval()
    x = torch.from_numpy(image[None].copy()).permute(0, 3, 1, 2).to(device)
    maps = model.features(model_input(x))
    logits = model.score(maps)
    prediction = int(logits.argmax(1).item())
    gradient, = torch.autograd.grad(logits[0, prediction], maps)
    cam = (gradient.mean(dim=(2, 3), keepdim=True) * maps).sum(1).relu().detach()
    cam = nn.functional.interpolate(cam[:, None], (SIZE, SIZE), mode="bilinear",
                                    align_corners=False)[0, 0].cpu().numpy()
    if cam.max() > 0:
        cam /= cam.max()
    return prediction, cam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--attention", action="store_true", help="Enable squeeze-and-excitation")
    parser.add_argument("--small-se", action="store_true", help="Use depthwise separable CNN blocks with SE")
    parser.add_argument("--seed", type=int, default=42, help="Training seed; the data split always uses seed 42")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--out", type=Path, help="Output directory for this training run")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.out is None:
        folder = "results_small_se" if args.small_se else ("results_attention" if args.attention else "results")
        if args.seed != 42:
            folder += f"_seed{args.seed}"
        args.out = Path("glaucoma") / folder

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    images, labels, paths = load_data(args.data, args.manifest)
    indices = np.arange(len(labels))
    train, held = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
    val, test = train_test_split(held, test_size=0.5, stratify=labels[held], random_state=42)

    def loader(split, shuffle=False):
        tensors = torch.from_numpy(images[split]).permute(0, 3, 1, 2)
        return DataLoader(TensorDataset(tensors, torch.from_numpy(labels[split])),
                          batch_size=32, shuffle=shuffle)

    train_loader, val_loader, test_loader = loader(train, True), loader(val), loader(test)
    model = GlaucomaCNN(attention=args.attention or args.small_se, separable=args.small_se).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    weights = len(train) / (2 * np.bincount(labels[train], minlength=2))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    best_auc = -1
    print(f"Images: {len(labels)}. Train/validation/test: {len(train)}/{len(val)}/{len(test)}. Device: {device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch, target in train_loader:
            batch, target = model_input(batch.to(device)), target.to(device)
            if torch.rand(()).item() < 0.5:
                batch = batch.flip(-1)
            optimizer.zero_grad()
            loss = criterion(model(batch), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
        truth, scores = evaluate(model, val_loader, device)
        auc = float(roc_auc_score(truth, scores))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train), "val_auc": auc})
        print(f"Epoch {epoch:02d} | train loss {history[-1]['train_loss']:.3f} | val AUROC {auc:.3f}", flush=True)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    torch.save(best_weights, args.out / "glaucoma_cnn.pt")
    model.load_state_dict(best_weights)
    truth, scores = evaluate(model, test_loader, device)
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    metrics = {
        "architecture": "small_se" if args.small_se else ("cnn_se" if args.attention else "cnn"),
        "seed": args.seed,
        "split_counts_normal_glaucoma": {
            name: np.bincount(labels[idx], minlength=2).tolist()
            for name, idx in (("train", train), ("validation", val), ("test", test))
        },
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch, "validation_auc": best_auc,
        "test_accuracy": float((truth == predictions).mean()),
        "test_balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "test_auc": float(roc_auc_score(truth, scores)),
        "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "history": history,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.out / "test_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "label", "glaucoma_score", "prediction"])
        for idx, score, prediction in zip(test, scores, predictions):
            writer.writerow([paths[idx].relative_to(args.data), labels[idx], score, prediction])

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.2))
    for row, label in enumerate(range(2)):
        # One example per ground-truth class, selected by split order.
        idx = next(idx for idx in test if labels[idx] == label)
        pred, cam = explain(model, images[idx], device)
        for ax in axes[row]:
            ax.imshow(images[idx])
            ax.axis("off")
        axes[row, 0].set_title(f"Actual {CLASSES[label]}")
        axes[row, 1].set_title(f"Predicted {CLASSES[pred]}")
        axes[row, 1].imshow(cam, cmap="inferno", vmin=0, vmax=1, alpha=0.65 * cam)
    fig.tight_layout()
    fig.savefig(args.out / "gradcam_examples.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
