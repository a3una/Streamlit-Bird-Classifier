"""Reusable CPU inference logic for a saved COS30082 classifier checkpoint."""
from pathlib import Path
import csv
import re

import torch
from PIL import Image

from src.models import create_model
from src.data import get_transforms


def load_classifier(checkpoint_path: Path):
    """Load the same architecture and preprocessing used during evaluation."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, _ = create_model(
        checkpoint["model_name"], int(checkpoint["num_classes"]), pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, preprocess = get_transforms(int(checkpoint["image_size"]))
    return model, preprocess, checkpoint


def species_names(checkpoint: dict, name_file: Path):
    """Use real labels when provided; otherwise show honest class IDs.

    Filenames such as 0237.jpg were originally used as class display names
    in some datasets. Those stems are not actual species names.
    """
    count = int(checkpoint["num_classes"])
    labels = [f"Class {idx}" for idx in range(count)]
    for idx, original in enumerate(checkpoint.get("class_names", [])):
        if idx >= count:
            break
        original = str(original).strip()
        # Numeric JPEG stems, e.g., 0237, are not meaningful species names.
        if original and not re.fullmatch(r"\d+|class\s*\d+", original, re.IGNORECASE):
            labels[idx] = original
    if name_file.is_file():
        with name_file.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                idx = int(row["class_id"])
                if 0 <= idx < count and row["species_name"].strip():
                    labels[idx] = row["species_name"].strip()
    return labels


def predict(image: Image.Image, model, preprocess, labels, top_k: int = 5):
    """Return (class ID, display name, probability) predictions."""
    if image is None:
        return []
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    scores, indices = torch.topk(probabilities, min(top_k, len(labels)))
    return [
        (int(idx), labels[int(idx)], float(score))
        for idx, score in zip(indices.tolist(), scores.tolist())
    ]
