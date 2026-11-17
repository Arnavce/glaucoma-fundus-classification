"""Extract the labeled images used in the report from local dataset ZIPs."""

import argparse
from collections import Counter
from pathlib import Path, PurePosixPath
import re
import shutil
from zipfile import ZipFile

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PREFIXES = {
    "ACRIMA/ACRIMA/NOT PARTITIONED": 705,
    "DRISHTI-GS/DRISHTI-GS": 101,
    "RIM-ONE/RIM-ONE/NOT PARTITIONED": 485,
    "ORIGA/ORIGA": 650,
}


def copy_member(archive, info, relative, destination):
    target = (destination / relative).resolve()
    if not target.is_relative_to(destination):
        raise ValueError(f"Unsafe archive path: {info.filename}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as incoming, target.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)


def extract_combined(path, destination):
    with ZipFile(path) as archive:
        counts = Counter()
        selected = []
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if info.is_dir() or name.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            for prefix in PREFIXES:
                if info.filename.startswith(prefix + "/"):
                    counts[prefix] += 1
                    selected.append((info, name))
                    break
        if counts != PREFIXES:
            raise ValueError(f"Unexpected combined archive layout or image counts: {dict(counts)}")
        for info, name in selected:
            copy_member(archive, info, name, destination)
    print(f"Extracted {sum(counts.values())} combined-archive images.")


def extract_refuge(path, destination):
    with ZipFile(path) as archive:
        counts = Counter()
        selected = []
        for info in archive.infolist():
            match = re.fullmatch(r"REFUGE2/train/images/([gn])([0-9]{4})\.jpg", info.filename)
            if match:
                prefix, number = match.groups()
                label = {"g": "glaucoma", "n": "normal"}[prefix]
                counts[label] += 1
                selected.append((info, Path("refuge_kaggle_labeled") / label / f"{prefix}{number}.jpg"))
        if counts != {"glaucoma": 40, "normal": 360}:
            raise ValueError(f"Unexpected REFUGE archive layout or image counts: {dict(counts)}")
        for info, name in selected:
            copy_member(archive, info, name, destination)
    print("Extracted 400 filename-labeled REFUGE images.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path)
    parser.add_argument("--refuge", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/glaucoma"))
    args = parser.parse_args()
    destination = args.output.resolve()
    if not (args.combined or args.refuge):
        parser.error("Provide --combined, --refuge, or both.")
    if args.combined:
        extract_combined(args.combined, destination)
    if args.refuge:
        extract_refuge(args.refuge, destination)


if __name__ == "__main__":
    main()
