"""Train Small CNN from scratch; train the ResNet head, then fine-tune the network."""

import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import DamageDataset, train_transform
from .evaluate import predict_dataset
from .models import SmallCNN, build_resnet18


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")


def train_one_epoch(model, loader, optimizer, device, freeze_batch_norm=False):
    model.train()
    if freeze_batch_norm:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    total_loss = 0.0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def train_small_cnn(final_train, val_loader, output_path, settings, device, seed=42, batch_size=64, num_workers=0):
    device = torch.device(device)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Choose a new run path; checkpoint already exists: {output_path}")
    if (
        "split" in val_loader.dataset.table
        and not val_loader.dataset.table["split"].eq("val").all()
    ):
        raise ValueError("Checkpoint selection requires validation data")
    train_sources = set(final_train["source_id"])
    if train_sources & set(val_loader.dataset.table["source_id"]):
        raise ValueError("Training and validation source groups overlap")
    set_seed(seed)
    model = SmallCNN().to(device)
    if device.type != "cuda":
        print("CUDA is not available. Small CNN training will continue on CPU and may be slow.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        DamageDataset(final_train, train_transform()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["max_epochs"])
    best_score, best_state, best_epoch = -1.0, None, None
    wait = 0
    history_rows = []
    started = time.perf_counter()

    for epoch in range(1, settings["max_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, labels, probabilities, _ = predict_dataset(model, val_loader, device)
        score = f1_score(labels, probabilities >= 0.5, average="macro")
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1_at_0.5": score,
            }
        )
        print(f"epoch {epoch:02d} | train loss {train_loss:.4f} | val macro-F1 {score:.4f}")
        if score > best_score + settings["min_delta"]:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        scheduler.step()
        if epoch >= settings["min_epochs"] and wait >= settings["patience"]:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
            break

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    metadata = {
        "model_name": "small_cnn",
        "threshold": 0.5,
        "seed": seed,
        "settings": settings,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "selected_epoch": best_epoch,
        "validation_macro_f1": float(best_score),
        "training_seconds": training_seconds,
    }
    torch.save({"model": best_state, **metadata}, output_path)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved the Small CNN model to {output_path}")
    history.to_csv(output_path.with_suffix(".history.csv"), index=False)
    return model, history


def train_resnet18(final_train, val_loader, output_path, settings, device, seed=42, batch_size=64, num_workers=0):
    device = torch.device(device)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Choose a new run path; checkpoint already exists: {output_path}")
    if (
        "split" in val_loader.dataset.table
        and not val_loader.dataset.table["split"].eq("val").all()
    ):
        raise ValueError("Checkpoint selection requires validation data")
    train_sources = set(final_train["source_id"])
    if train_sources & set(val_loader.dataset.table["source_id"]):
        raise ValueError("Training and validation source groups overlap")
    set_seed(seed)
    model = build_resnet18(pretrained=True).to(device)
    if device.type != "cuda":
        print("CUDA is not available. ResNet-18 fine-tuning will continue on CPU and may be slow.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        DamageDataset(final_train, train_transform()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.01)
    best_score, best_state, best_stage, best_epoch = -1.0, None, None, None
    wait = 0
    history_rows = []
    started = time.perf_counter()

    for epoch in range(1, settings["head_max_epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, freeze_batch_norm=True
        )
        val_loss, labels, probabilities, _ = predict_dataset(model, val_loader, device)
        score = f1_score(labels, probabilities >= 0.5, average="macro")
        history_rows.append(
            {
                "stage": "linear_probe",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1_at_0.5": score,
            }
        )
        print(f"linear probe {epoch:02d} | train loss {train_loss:.4f} | val macro-F1 {score:.4f}")
        if score > best_score + settings["min_delta"]:
            best_score, best_state = score, deepcopy(model.state_dict())
            best_stage, best_epoch = "linear_probe", epoch
            wait = 0
        else:
            wait += 1
        if epoch >= settings["head_min_epochs"] and wait >= settings["head_patience"]:
            print(f"Linear-probe early stopping after epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    for parameter in model.parameters():
        parameter.requires_grad = True
    backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": 0.0001},
            {"params": model.fc.parameters(), "lr": 0.0003},
        ],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["finetune_max_epochs"]
    )
    wait = 0

    for epoch in range(1, settings["finetune_max_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, labels, probabilities, _ = predict_dataset(model, val_loader, device)
        score = f1_score(labels, probabilities >= 0.5, average="macro")
        history_rows.append(
            {
                "stage": "fine_tune",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1_at_0.5": score,
            }
        )
        print(f"fine tune   {epoch:02d} | train loss {train_loss:.4f} | val macro-F1 {score:.4f}")
        if score > best_score + settings["min_delta"]:
            best_score, best_state = score, deepcopy(model.state_dict())
            best_stage, best_epoch = "fine_tune", epoch
            wait = 0
        else:
            wait += 1
        scheduler.step()
        if epoch >= settings["finetune_min_epochs"] and wait >= settings["finetune_patience"]:
            print(
                f"Fine-tuning stopped after epoch {epoch}; "
                f"best checkpoint: {best_stage}, epoch {best_epoch}."
            )
            break

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    metadata = {
        "model_name": "resnet18",
        "threshold": 0.5,
        "seed": seed,
        "settings": settings,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "selected_stage": best_stage,
        "selected_epoch": best_epoch,
        "validation_macro_f1": float(best_score),
        "training_seconds": training_seconds,
    }
    torch.save({"model": best_state, **metadata}, output_path)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved the ResNet-18 model to {output_path}")
    history.to_csv(output_path.with_suffix(".history.csv"), index=False)
    return model, history
