"""Read data, check source splits, and prepare training slices and rotations."""

import math
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def load_splits(data_dir, manifest_path):
    """Check metadata without opening validation/test image pixels."""
    data_dir = Path(data_dir)
    labels = pd.read_csv(data_dir / "labels.csv")
    split = pd.read_csv(manifest_path)
    if labels["name"].duplicated().any() or split["name"].duplicated().any():
        raise ValueError("Duplicate label/split names require an explicit review")
    if not labels["label"].isin([0, 1]).all():
        raise ValueError("Expected binary labels 0 and 1")
    paths = sorted(
        p
        for p in data_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} and p.is_file()
    )
    image_names = {p.name for p in paths}
    label_names = set(labels["name"])
    audit = {
        "label_rows": len(labels),
        "image_files": len(paths),
        "matched_samples": len(label_names & image_names),
        "labels_without_image": sorted(label_names - image_names),
        "images_without_label": sorted(image_names - label_names),
        "policy": "List unmatched records; review corrupt files and duplicates before training.",
    }
    if set(split["name"]) != label_names & image_names:
        raise ValueError("Fixed manifest does not cover exactly the matched images")
    if not split["name"].str.contains("_crop_", regex=False).all():
        raise ValueError("Cannot derive source_id from a non-crop filename")
    derived = split["name"].str.split("_crop_").str[0]
    if not derived.equals(split["source_id"]):
        raise ValueError("source_id must be the prefix before _crop_")
    if split.groupby("source_id")["split"].nunique().max() != 1:
        raise ValueError("A source image appears in multiple splits")
    if set(split["split"]) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test splits")
    original_labels = labels.set_index("name")["label"]
    if not split["name"].map(original_labels).equals(split["label"]):
        raise ValueError("Manifest labels disagree with labels.csv")
    split["path"] = split["name"].map(lambda name: data_dir / name)
    train_df = split[split["split"] == "train"].reset_index(drop=True)
    val_df = split[split["split"] == "val"].reset_index(drop=True)
    test_df = split[split["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df, audit


def audit_images(table):
    """Inspect only the supplied rows; return all corrupt and duplicate records."""
    rows, corrupt = [], []
    for row in table.itertuples(index=False):
        path = Path(row.path)
        try:
            with Image.open(path) as image:
                image.verify()
            rows.append(
                {
                    "name": row.name,
                    "label": row.label,
                    "source_id": row.source_id,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
        except (OSError, ValueError) as error:
            corrupt.append({"name": row.name, "error": str(error)})
    hashes = pd.DataFrame(rows, columns=["name", "label", "source_id", "sha256"])
    duplicates = hashes[hashes.duplicated("sha256", keep=False)]
    return {
        "checked_images": len(table),
        "corrupt": corrupt,
        "duplicates": duplicates.to_dict("records"),
        "policy": "Do not silently remove rows; resolve reported problems before training.",
    }


class DamageDataset(Dataset):
    def __init__(self, table, transform):
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        if "box" in self.table:
            image = render_crop(row["path"], row["box"], row["rotation_degrees"])
            tensor = self.transform(image)
        else:
            with Image.open(row["path"]) as image:
                tensor = self.transform(image)
        return tensor, int(row["label"]), row["name"]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resize_and_pad(image, image_size=224):
    # Keep small images at their original size, matching the saved models.
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (128, 128, 128))
    canvas.paste(image, ((image_size - image.width) // 2, (image_size - image.height) // 2))
    return canvas


def train_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluation_transform(image_size=224):
    return transforms.Compose(
        [
            partial(resize_and_pad, image_size=image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def segment_bounds(long_side, short_side):
    count = max(2, math.ceil(long_side / short_side / 2.0))
    edges = np.linspace(0, long_side, count + 1).round().astype(int)
    overlap = round(short_side * 0.10)
    return [
        (
            max(0, int(edges[i]) - (overlap if i else 0)),
            min(long_side, int(edges[i + 1]) + (overlap if i + 1 < count else 0)),
        )
        for i in range(count)
    ]


def stable_choice(key, choices, seed=42):
    digest = sha256(f"{seed}:{key}".encode()).digest()
    return choices[int.from_bytes(digest[:8], "big") % len(choices)]


def prepare_training_rows(train_df, review_path):
    """Build crop coordinates first, without writing thousands of images."""
    if not train_df["split"].eq("train").all():
        raise ValueError("Only training rows may be sliced or augmented")
    review = pd.read_csv(review_path)
    if review.duplicated(["image_name", "slice_index"]).any():
        raise ValueError("Duplicate slice review records")
    if not review["keep"].isin([0, 1]).all():
        raise ValueError("Each slice needs keep=0 or keep=1")
    decisions = review.set_index(["image_name", "slice_index"])["keep"].to_dict()
    originals, candidates = [], []
    elongated = 0
    for row in train_df.itertuples(index=False):
        with Image.open(row.path) as image:
            width, height = image.size
            if image.getexif().get(274) in [5, 6, 7, 8]:
                width, height = height, width
        common = {
            "original_name": row.name,
            "path": row.path,
            "source_id": row.source_id,
            "label": int(row.label),
            "split": "train",
        }
        originals.append(
            {
                **common,
                "base_id": f"{Path(row.name).stem}__full",
                "base_kind": "original",
                "slice_index": 0,
                "box": (0, 0, width, height),
            }
        )
        long_side, short_side = max(width, height), min(width, height)
        if long_side / short_side < 2:
            continue
        elongated += 1
        bounds = segment_bounds(long_side, short_side)
        for i, (start, end) in enumerate(bounds, 1):
            box = (start, 0, end, height) if width >= height else (0, start, width, end)
            candidates.append(
                {
                    **common,
                    "base_id": f"{Path(row.name).stem}__slice_{i:02d}of{len(bounds):02d}",
                    "base_kind": "slice",
                    "slice_index": i,
                    "box": box,
                }
            )
    expected = {(r["original_name"], r["slice_index"]) for r in candidates}
    if set(decisions) != expected:
        raise ValueError("Slice review must cover exactly all training candidates")
    retained = [r for r in candidates if decisions[r["original_name"], r["slice_index"]] == 1]
    summary = {
        "original_train_images": len(originals),
        "elongated_originals": elongated,
        "candidate_slices": len(candidates),
        "accepted_slices": len(retained),
        "rejected_slices": len(candidates) - len(retained),
        "base_images": len(originals) + len(retained),
        "augmented_images": 3 * (len(originals) + len(retained)),
    }
    return pd.DataFrame(originals + retained), summary


def render_crop(image_path, box, angle=0, image_size=224):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB").crop(box)
    if angle:
        image = image.rotate(
            angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(128, 128, 128)
        )
    # Training can upscale crops; inference cannot. This matches the original notebook.
    image = ImageOps.contain(image, (image_size, image_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (128, 128, 128))
    canvas.paste(image, ((image_size - image.width) // 2, (image_size - image.height) // 2))
    return canvas


def augment_training_rows(base_table, seed=42):
    """Add deterministic rotation choices in memory; images are rendered by the Dataset."""
    if not base_table["split"].eq("train").all():
        raise ValueError("Only training data may be augmented")
    rows = []
    for base in base_table.to_dict("records"):
        angles = [
            0,
            stable_choice(f"{base['base_id']}:orthogonal", [90, 270], seed),
            stable_choice(f"{base['base_id']}:diagonal", [45, 135, 225, 315], seed),
        ]
        for angle in angles:
            rows.append({**base, "rotation_degrees": angle,
                         "name": f"{base['base_id']}__rotate_{angle:03d}"})
    return pd.DataFrame(rows)
