"""Check data splits, preprocessing, training and inference."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from src.data import (
    DamageDataset,
    audit_images,
    augment_training_rows,
    render_crop,
    train_transform,
    evaluation_transform,
    load_splits,
    prepare_training_rows,
    resize_and_pad,
)
from src.evaluate import metric_summary, predict_dataset, predict_image
from src.models import SmallCNN, build_resnet18, load_model
from src.train import train_resnet18, train_small_cnn


@pytest.fixture
def example_data(tmp_path):
    rows = []
    for i, part in enumerate(["train", "val", "test"]):
        for label in [0, 1]:
            name = f"image_{i}_{label}_crop_0001.png"
            Image.new("RGB", (96, 32), (20 + 70 * i, 30 + 80 * label, 100)).save(tmp_path / name)
            rows.append(
                {"name": name, "source_id": f"image_{i}_{label}", "label": label, "split": part}
            )
    table = pd.DataFrame(rows)
    table[["name", "label"]].to_csv(tmp_path / "labels.csv", index=False)
    table.to_csv(tmp_path / "split.csv", index=False)
    return tmp_path


def test_split_checks_and_unmatched_records(example_data):
    labels = pd.read_csv(example_data / "labels.csv")
    labels.loc[len(labels)] = ["missing_crop_1.png", 0]
    labels.to_csv(example_data / "labels.csv", index=False)
    Image.new("RGB", (20, 20)).save(example_data / "unlabelled.png")
    train, val, test, audit = load_splits(example_data, example_data / "split.csv")
    assert audit["labels_without_image"] == ["missing_crop_1.png"]
    assert audit["images_without_label"] == ["unlabelled.png"]
    assert len(train) == len(val) == len(test) == 2
    table = pd.read_csv(example_data / "split.csv")
    table.loc[0, "label"] = 1
    table.to_csv(example_data / "split.csv", index=False)
    with pytest.raises(ValueError, match="labels disagree"):
        load_splits(example_data, example_data / "split.csv")


def test_cross_split_source_is_rejected(example_data):
    table = pd.read_csv(example_data / "split.csv")
    old = table.loc[2, "name"]
    new = table.loc[0, "source_id"] + "_crop_0002.png"
    (example_data / old).rename(example_data / new)
    table.loc[2, ["name", "source_id"]] = [new, table.loc[0, "source_id"]]
    table.to_csv(example_data / "split.csv", index=False)
    table[["name", "label"]].to_csv(example_data / "labels.csv", index=False)
    with pytest.raises(ValueError, match="multiple splits"):
        load_splits(example_data, example_data / "split.csv")


def test_preprocessing_keeps_geometry_and_training_sources(example_data):
    raw = Image.new("RGB", (20, 10), "white")
    padded = np.asarray(resize_and_pad(raw))
    assert (padded[:, :, 0] == 255).sum() == 200
    assert raw.size == (20, 10)
    train, val, _, _ = load_splits(example_data, example_data / "split.csv")
    accepted = example_data / "review.csv"
    pd.DataFrame([{"image_name": name, "slice_index": i, "keep": 0}
                  for name in train.name for i in [1, 2]]).to_csv(accepted, index=False)
    bases, summary = prepare_training_rows(train, accepted)
    assert summary["candidate_slices"] == 4
    augmented = augment_training_rows(bases)
    assert len(augmented) == 6
    assert augmented.groupby("base_id").size().eq(3).all()
    assert set(augmented.source_id) == set(train.source_id)
    with pytest.raises(ValueError, match="Only training"):
        prepare_training_rows(val, accepted)
    before = set(example_data.iterdir())
    dataset = DamageDataset(augmented, train_transform())
    for i, row in augmented.iterrows():
        actual, label, name = dataset[i]
        expected = train_transform()(render_crop(row.path, row.box, row.rotation_degrees))
        assert torch.equal(actual, expected)
        assert actual.shape == (3, 224, 224) and label == row.label and name == row["name"]
    assert set(example_data.iterdir()) == before
    assert augmented.equals(augment_training_rows(bases, seed=42))
    review = pd.read_csv(accepted)
    for bad in [review.iloc[:-1], pd.concat([review, review.iloc[:1]]), review.assign(keep=2)]:
        bad.to_csv(accepted, index=False)
        with pytest.raises(ValueError):
            prepare_training_rows(train, accepted)


def test_corruption_and_duplicates_are_reported(example_data):
    train, _, _, _ = load_splits(example_data, example_data / "split.csv")
    train.iloc[1]["path"].write_bytes(train.iloc[0]["path"].read_bytes())
    assert len(audit_images(train)["duplicates"]) == 2
    train.iloc[0]["path"].write_bytes(b"corrupt")
    assert len(audit_images(train)["corrupt"]) == 1


def test_metrics_use_checkpoint_threshold_and_both_classes():
    metrics, matrix, predictions = metric_summary([0, 0, 1, 1], [0.1, 0.55, 0.6, 0.9], 0.6)
    assert predictions.tolist() == [0, 0, 1, 1]
    assert matrix.tolist() == [[2, 0], [0, 2]]
    assert metrics["macro_f1"] == metrics["crack_recall"] == 1
    with pytest.raises(ValueError):
        metric_summary([0, 1], [0.5, float("nan")])


@pytest.mark.parametrize("key", ["small_cnn", "resnet18"])
def test_checkpoint_loading(tmp_path, key):
    torch.set_num_threads(2)
    model = SmallCNN() if key == "small_cnn" else build_resnet18()
    assert sum(p.numel() for p in model.parameters()) == (
        1_874_914 if key == "small_cnn" else 11_177_538
    )
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "threshold": 0.62}, path)
    loaded, threshold = load_model(key, path)
    assert threshold == 0.62


@pytest.mark.parametrize("key", ["small_cnn", "resnet18"])
def test_training_and_inference_have_no_notebook_globals(example_data, monkeypatch, key):
    torch.set_num_threads(2)
    train, val, _, _ = load_splits(example_data, example_data / "split.csv")
    loader = DataLoader(DamageDataset(val, evaluation_transform()), batch_size=2)
    settings = {
        "max_epochs": 1,
        "min_epochs": 1,
        "patience": 1,
        "min_delta": 0.001,
        "head_max_epochs": 1,
        "head_min_epochs": 1,
        "head_patience": 1,
        "finetune_max_epochs": 1,
        "finetune_min_epochs": 1,
        "finetune_patience": 1,
    }
    monkeypatch.setattr("src.train.build_resnet18", lambda pretrained: build_resnet18())
    trainer = train_small_cnn if key == "small_cnn" else train_resnet18
    _, history = trainer(train, loader, example_data / f"{key}.pt", settings, "cpu")
    assert len(history) == (1 if key == "small_cnn" else 2)
    loaded, threshold = load_model(key, example_data / f"{key}.pt")
    _, _, probabilities, names = predict_dataset(loaded, loader, torch.device("cpu"))
    assert names == val.name.tolist()
    one = predict_image(loaded, val.iloc[0].path, threshold)
    assert abs(one["p_crack"] - probabilities[0]) < 1e-5
    assert json.loads((example_data / f"{key}.json").read_text())["selected_epoch"] == 1


def test_kept_slices_follow_exif_orientation(tmp_path):
    pixels = np.zeros((32, 96, 3), dtype=np.uint8)
    pixels[:, :48, 0] = 255
    pixels[:, 48:, 2] = 255
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "source_crop_1.png"
    image.save(path, exif=exif)
    train = pd.DataFrame([{"name": path.name, "path": path, "source_id": "source",
                           "label": 1, "split": "train"}])
    review = tmp_path / "review.csv"
    pd.DataFrame({"image_name": [path.name, path.name], "slice_index": [1, 2],
                  "keep": [1, 0]}).to_csv(review, index=False)
    bases, summary = prepare_training_rows(train, review)
    assert summary["accepted_slices"] == 1
    sliced = bases[bases.base_kind.eq("slice")].iloc[0]
    assert sliced.box == (0, 0, 32, 51)
    assert sliced.slice_index == 1
    with Image.open(path) as original:
        expected = ImageOps.exif_transpose(original).convert("RGB").crop(sliced.box)
    expected = ImageOps.contain(expected, (224, 224), Image.Resampling.BILINEAR)
    rendered = render_crop(path, sliced.box)
    left = (224 - expected.width) // 2
    top = (224 - expected.height) // 2
    assert np.array_equal(np.asarray(rendered.crop((left, top, left + expected.width,
                                                   top + expected.height))), np.asarray(expected))


def test_image_factors_and_source_class_weights(tmp_path):
    from src.features import extract_simple_features, source_class_weights

    for size in [(1, 1), (1, 200), (200, 1), (80, 120)]:
        path = tmp_path / 'image.png'
        Image.new('RGB', size, (160, 80, 40)).save(path)
        values = extract_simple_features(path)
        assert len(values) == 49 and np.isfinite(list(values.values())).all()
        assert values['red_mean'] == pytest.approx(160 / 255)
        assert values['edge_mean'] == pytest.approx(0)
    table = pd.DataFrame({'source_id': ['a', 'a', 'b', 'c', 'd', 'd', 'd'],
                          'label': [0, 0, 0, 1, 1, 1, 1]})
    weights = source_class_weights(table)
    assert weights[table.label.eq(0)].sum() == pytest.approx(weights[table.label.eq(1)].sum())
    assert weights[:2].sum() == pytest.approx(weights[2])
    assert weights[3] == pytest.approx(weights[4:].sum())


def test_lgbm_saved_model_predicts_without_feature_cache(tmp_path):
    import lightgbm as lgb
    from src.features import extract_simple_features, predict_lgbm

    rows = []
    for i in range(40):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (40, 30), (30 + i * 4, 90, 120)).save(path)
        rows.append(extract_simple_features(path))
    features = pd.DataFrame(rows)
    model = lgb.LGBMClassifier(n_estimators=5, min_child_samples=2, verbosity=-1, n_jobs=1)
    model.fit(features, np.arange(40) >= 20)
    (tmp_path / 'lightgbm.txt').write_text(model.booster_.model_to_string(), encoding='utf-8')
    restored = lgb.Booster(model_str=(tmp_path / 'lightgbm.txt').read_text(encoding='utf-8'))
    actual = predict_lgbm(restored, path)
    assert actual['p_crack'] == pytest.approx(model.predict_proba(features.iloc[-1:])[0, 1])



def test_resnet_gradcam_cleans_hooks_and_keeps_predictions():
    from src.gradcam import grad_cam

    torch.set_num_threads(2)
    model = build_resnet18().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    inputs = torch.randn(1, 3, 64, 64)
    before = model(inputs).detach()
    for label in (0, 1):
        heatmap = grad_cam(model, model.layer4[-1], inputs, label)
        assert heatmap.shape == (64, 64) and np.isfinite(heatmap).all()
        assert 0 <= heatmap.min() <= heatmap.max() <= 1
        assert not model.layer4[-1]._forward_hooks
    with pytest.raises(IndexError):
        grad_cam(model, model.layer4[-1], inputs, 3)
    assert not model.layer4[-1]._forward_hooks
    assert torch.equal(before, model(inputs))


def test_single_feature_threshold_matches_training_objective():
    from src.features import fit_feature_threshold
    from sklearn.metrics import f1_score, accuracy_score

    values = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8])
    labels = np.array([1, 1, 0, 1, 0, 0, 0])
    rule = fit_feature_threshold(values, labels)
    assert rule['direction'] == '<='
    thresholds = np.unique(np.round(np.r_[values.min()-0.0001,
        (values[:-1]+values[1:])/2, values.max()+0.0001], 4))
    midpoint = np.mean([values[labels == label].mean() for label in (0, 1)])
    best = max(thresholds, key=lambda t: (f1_score(labels, values <= t, average='macro'),
        accuracy_score(labels, values <= t), -abs(t-midpoint), -t))
    assert rule['threshold'] == best
    assert fit_feature_threshold(-values, labels)['direction'] == '>='
