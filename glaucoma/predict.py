"""Predict a class and optionally save a Grad-CAM overlay for one image."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from train import CLASSES, GlaucomaCNN, explain, model_input, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=("baseline", "standard-se", "small-se"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--gradcam", type=Path, help="PNG path for the optional Grad-CAM overlay")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = GlaucomaCNN(attention=args.architecture != "baseline",
                        separable=args.architecture == "small-se").eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as error:
        parser.error(f"The checkpoint does not match architecture {args.architecture}: {error}")
    image = preprocess(args.image)
    inp = torch.from_numpy(image.copy()).permute(2, 0, 1)[None]
    with torch.inference_mode():
        score = float(model(model_input(inp)).softmax(1)[0, 1])
    prediction = int(score >= 0.5)
    print(f"{CLASSES[prediction]} | glaucoma score: {score:.4f}")
    if args.gradcam:
        _, heatmap = explain(model, image, "cpu")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.imshow(heatmap, cmap="inferno", vmin=0, vmax=1, alpha=.65 * heatmap)
        ax.set_title(CLASSES[prediction])
        ax.axis("off")
        fig.savefig(args.gradcam, dpi=160, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
