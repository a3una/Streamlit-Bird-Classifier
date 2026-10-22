"""Dataset readers and transforms for the CUB-200 assignment.

This accepts Train.zip / Test.zip *without extraction*, or folders Train/ / Test/.
Annotation lines must contain: <filename> <zero-based integer label>
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import random
import re
import zipfile

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Record:
    filename: str
    label: int


def load_annotations(data_dir: Path, split: str) -> list[Record]:
    """Read train.txt or test.txt and check for duplicate filenames."""
    path = data_dir / f"{split.lower()}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    records = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                filename, label_str = line.rsplit(maxsplit=1)
                label = int(label_str)
            except ValueError as exc:
                raise ValueError(f"Invalid annotation at {path}:{line_no}: {line!r}") from exc
            if label < 0:
                raise ValueError(f"Negative class label at {path}:{line_no}")
            key = filename.replace("\\", "/").lower()
            if key in seen:
                raise ValueError(f"Duplicate image annotation in {path}: {filename}")
            seen.add(key)
            records.append(Record(filename, label))
    if not records:
        raise ValueError(f"No annotations found in {path}")
    return records


class ImageSource:
    """Find images in a folder or ZIP and read only one image at a time.

    ZIP handles are opened lazily per DataLoader process, avoiding an all-in-memory
    dataset and permitting Windows' multiprocessing pickling.
    """
    def __init__(self, data_dir: Path, split: str):
        data_dir = Path(data_dir)
        split = split.lower()
        directory_candidates = [data_dir / split, data_dir / split.capitalize()]
        archive_candidates = [data_dir / f"{split}.zip", data_dir / f"{split.capitalize()}.zip"]
        self.root = next((p for p in directory_candidates if p.is_dir()), None)
        self.archive = None if self.root else next((p for p in archive_candidates if p.is_file()), None)
        if self.root is None and self.archive is None:
            raise FileNotFoundError(
                f"Cannot find {split} images in {data_dir}. Put {split.capitalize()}.zip "
                f"or an extracted {split.capitalize()}/ folder here."
            )
        self._zip = None
        self._images = {}
        self._basenames = {}
        if self.root is not None:
            paths = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            for p in paths:
                key = p.relative_to(self.root).as_posix().lower()
                self._images[key] = p
                self._add_basename(p.name.lower(), key)
        else:
            with zipfile.ZipFile(self.archive) as archive:
                for name in archive.namelist():
                    if Path(name).suffix.lower() in IMAGE_EXTS and not name.startswith("__MACOSX/"):
                        key = name.replace("\\", "/").lower()
                        self._images[key] = name
                        self._add_basename(Path(name).name.lower(), key)
        if not self._images:
            raise ValueError(f"No image files detected in {self.root or self.archive}")

    def _add_basename(self, basename: str, key: str) -> None:
        # A duplicate basename is intentionally ambiguous; do not silently choose.
        if basename not in self._basenames:
            self._basenames[basename] = key
        elif self._basenames[basename] != key:
            self._basenames[basename] = None

    def _resolve(self, filename: str):
        normalized = filename.replace("\\", "/").lstrip("./").lower()
        if normalized in self._images:
            return self._images[normalized]
        basename = normalized.rsplit("/", 1)[-1]
        key = self._basenames.get(basename)
        if key is None:
            raise FileNotFoundError(
                f"Image {filename!r} is missing or has an ambiguous basename "
                f"in {self.root or self.archive}."
            )
        return self._images[key]

    def check_records(self, records: list[Record]) -> None:
        """Fail early if annotations refer to missing/ambiguous files."""
        missing = []
        for record in records:
            try:
                self._resolve(record.filename)
            except FileNotFoundError:
                missing.append(record.filename)
                if len(missing) == 5:
                    break
        if missing:
            raise FileNotFoundError(f"Cannot locate annotated images, e.g. {missing}")

    def read(self, filename: str) -> Image.Image:
        location = self._resolve(filename)
        if self.root is not None:
            with Image.open(location) as image:
                return image.convert("RGB")
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive, "r")
        data = self._zip.read(location)
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_zip"] = None
        return state

    def __del__(self):
        try:
            if self._zip is not None:
                self._zip.close()
        except AttributeError:
            pass


class BirdDataset(Dataset):
    def __init__(self, records: list[Record], source: ImageSource, transform):
        self.records = records
        self.source = source
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = self.source.read(record.filename)
        return self.transform(image), record.label


def split_stratified(records: list[Record], validation_fraction: float = 0.2, seed: int = 42):
    """Create a stratified validation set *only* from the training annotations."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    by_class = defaultdict(list)
    for record in records:
        by_class[record.label].append(record)
    rng = random.Random(seed)
    train, val = [], []
    for label, items in sorted(by_class.items()):
        items = list(items)
        rng.shuffle(items)
        if len(items) < 2:
            raise ValueError(f"Class {label} has fewer than 2 images: cannot stratify train/validation")
        val_count = min(len(items) - 1, max(1, round(len(items) * validation_fraction)))
        val.extend(items[:val_count])
        train.extend(items[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def extract_species_name(filename: str) -> str:
    """Convert CUB filenames such as Black_footed_Albatross_0019_416160254.jpg."""
    stem = Path(filename.replace("\\", "/")).stem
    match = re.match(r"^(.*?)_\d{4}_.+$", stem)
    name = match.group(1) if match else stem
    return name.replace("_", " ").strip()


def get_class_names(records: list[Record], num_classes: int) -> list[str]:
    names = [f"Class {i}" for i in range(num_classes)]
    for record in records:
        if 0 <= record.label < num_classes:
            names[record.label] = extract_species_name(record.filename)
    # Prefix class IDs only if names are duplicate; output keys must be distinct.
    if len(set(names)) != len(names):
        names = [f"{i}: {name}" for i, name in enumerate(names)]
    return names


def get_transforms(image_size: int = 224):
    """Use data augmentation for training, deterministic preprocessing for evaluation."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(round(image_size * 256 / 224), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
    ])
    return train_transform, eval_transform
