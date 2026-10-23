"""
Reproducible Training Pipeline for VSL Continuous Recognition Model & Single-Sign Baseline.

Implements:
1. Primary Continuous CTC Training (`train_ctc_pipeline`):
   - Architecture: `VSLSignCTCModel` (`[B, T, 212]` -> Conv1D + 2-layer BiLSTM + Linear + LogSoftmax)
   - Loss: `torch.nn.CTCLoss(blank=0, zero_infinity=True)` with `[T, B, C]` log-probabilities
   - Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
   - Gradient Clipping: `clip_grad_norm_(max_norm=2.0)`
   - Scheduler: `CosineAnnealingLR(T_max=epochs)`
   - Strict group-disjoint train/val/test splitting (`speakerId` -> `sessionId` -> `recordingId`)
   - Validation tracking: Loss, Gloss WER, CER, Sequence Exact-Match, S/D/I/H edit breakdown
   - Best-checkpoint selection strictly using validation performance (never test set)
   - Configurable early stopping
   - Self-describing `.pth` and `.meta.json` checkpoint export
2. Secondary Single-Sign Classification Baseline Training (`train_single_sign_pipeline`):
   - Architecture: `VSLSingleSignBaselineModel` (`[B, 30, 138]` -> Conv1D + MaxPool1D + BiLSTM + Dense)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

from dataset import (
    VSLContinuousDataset,
    load_single_sign_rows,
    pad_collate_fn,
    parse_and_validate_samples,
    split_dataset_by_group,
    split_single_sign_dataset,
)
from decoder import BLANK_ID, BLANK_TOKEN, Vocabulary
from evaluate import _collate_with_meta, evaluate_ctc_model, evaluate_single_sign_model
from feature_extractor import (
    CTC_FEATURE_DIM,
    FEATURE_VERSION,
    SINGLE_SIGN_FEATURE_DIM,
    SINGLE_SIGN_SEQUENCE_LENGTH,
)
from model import VSLSignCTCModel, VSLSingleSignBaselineModel


def set_seed(seed: int) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML configuration if path exists, otherwise return empty dict."""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def train_ctc_pipeline(
    raw_samples: Sequence[Dict[str, Any]],
    output_model_path: str,
    vocab: Optional[Vocabulary] = None,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 2.0,
    seed: int = 42,
    early_stopping_patience: int = 15,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """
    Train `VSLSignCTCModel` on validated continuous VSL landmark/feature sequences.
    """
    set_seed(seed)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    if vocab is None:
        vocab = Vocabulary()

    valid_samples, skipped_count, diag = parse_and_validate_samples(
        raw_samples, vocab=vocab, return_diagnostics=True
    )
    if len(valid_samples) == 0:
        raise ValueError("No valid continuous CTC samples remaining after validation.")

    train_samples, val_samples, test_samples = split_dataset_by_group(
        valid_samples, seed=seed
    )

    train_dataset = VSLContinuousDataset(train_samples)
    val_dataset = VSLContinuousDataset(val_samples)
    test_dataset = VSLContinuousDataset(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_with_meta,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_with_meta,
    )

    model = VSLSignCTCModel(
        input_dim=CTC_FEATURE_DIM,
        num_classes=vocab.num_classes,
    ).to(device)

    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=max(1e-4, lr * 0.1)
    )

    best_val_wer = float("inf")
    best_val_loss = float("inf")
    best_val_non_blank = False
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    meta_path = (
        output_model_path[:-4] + ".meta.json"
        if output_model_path.endswith(".pth")
        else output_model_path + ".meta.json"
    )

    training_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "seed": seed,
        "early_stopping_patience": early_stopping_patience,
        "device": str(device),
    }

    split_summary = {
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_speakers": sorted({s["speaker_id"] for s in train_samples}),
        "val_speakers": sorted({s["speaker_id"] for s in val_samples}),
        "test_speakers": sorted({s["speaker_id"] for s in test_samples}),
    }

    diag_dict = asdict(diag)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        valid_batches = 0
        failed_batches = 0

        for features, targets, input_lengths, target_lengths, _ in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(features, input_lengths)  # [B, T, C]
            log_probs_tbc = log_probs.transpose(0, 1)  # [T, B, C]

            loss = criterion(log_probs_tbc, targets, input_lengths, target_lengths)
            if not math.isfinite(loss.item()):
                failed_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            train_loss_sum += float(loss.item())
            valid_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / float(max(1, valid_batches))

        val_metrics = evaluate_ctc_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            vocab=vocab,
            device=device,
            decode_method="greedy",
        )
        val_loss = float(val_metrics["loss"])
        val_wer = float(val_metrics["corpus_wer"])
        val_cer = float(val_metrics["mean_sample_cer"])
        val_exact = float(val_metrics["exact_match_rate"])

        epoch_record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_wer": val_wer,
            "val_cer": val_cer,
            "val_exact_match_rate": val_exact,
            "val_substitutions": int(val_metrics["substitutions"]),
            "val_deletions": int(val_metrics["deletions"]),
            "val_insertions": int(val_metrics["insertions"]),
            "val_hits": int(val_metrics["hits"]),
            "failed_batches": failed_batches,
        }
        history.append(epoch_record)

        # Best checkpoint selection on validation performance (never lock onto degenerate all-blank hits=0 over hits>0)
        val_hits = int(val_metrics["hits"])
        cur_non_blank = (
            val_hits > 0
            or int(val_metrics["substitutions"]) > 0
            or int(val_metrics["insertions"]) > 0
        )
        if best_epoch == 0:
            is_better = True
        elif cur_non_blank and not best_val_non_blank:
            is_better = True
        elif not cur_non_blank and best_val_non_blank:
            is_better = False
        else:
            is_better = (val_wer < best_val_wer - 1e-6) or (
                abs(val_wer - best_val_wer) <= 1e-6 and val_loss < best_val_loss
            )

        if is_better:
            best_val_wer = val_wer
            best_val_loss = val_loss
            best_val_non_blank = cur_non_blank
            best_epoch = epoch
            epochs_without_improvement = 0

            vocab_info = vocab.to_dict()
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "architecture": model.get_architecture_config(),
                "feature_dim": CTC_FEATURE_DIM,
                "feature_version": FEATURE_VERSION,
                "vocabulary": [vocab.id_to_word[i] for i in range(vocab.num_classes)],
                "gloss_to_id": vocab.word_to_id,
                "id_to_gloss": vocab_info["idToWord"],
                "num_classes": vocab.num_classes,
                "blank_id": BLANK_ID,
                "blank_token": BLANK_TOKEN,
                "training_config": training_config,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_wer": best_val_wer,
                "val_metrics": {k: v for k, v in val_metrics.items() if k != "predictions"},
                "seed": seed,
                "dataset_diagnostics": diag_dict,
                "split_summary": split_summary,
            }
            torch.save(checkpoint_payload, output_model_path)

            meta_path_out = meta_path
            meta_payload = {
                k: v for k, v in checkpoint_payload.items() if k != "model_state_dict"
            }
            with open(meta_path_out, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=2, ensure_ascii=False)
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            break

    best_ckpt = torch.load(output_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate_ctc_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        vocab=vocab,
        device=device,
        decode_method="greedy",
    )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_wer": best_val_wer,
        "checkpoint_path": output_model_path,
        "metadata_path": meta_path,
        "split_summary": split_summary,
        "dataset_diagnostics": diag_dict,
        "history": history,
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "predictions"},
    }


def train_single_sign_pipeline(
    rows: Sequence[Dict[str, Any]],
    output_model_path: str,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """
    Train the secondary deliverable `VSLSingleSignBaselineModel` (30 frames x 138 features).
    Accepts either `DatasetRow` records (`x_0..z_20, label, sequence_id`) or pre-extracted
    `{"label": ..., "sequence": [[138 floats] x 30], "speakerId": ..., "split": ...}` dicts.
    """
    set_seed(seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    # Support both pre-extracted 30x138 sequence dicts and raw landmark DatasetRow dicts
    if len(rows) > 0 and "sequence" in rows[0]:
        from decoder import normalize_single_gloss
        class_names = sorted({normalize_single_gloss(r["label"]) for r in rows})
        cls_to_idx = {c: i for i, c in enumerate(class_names)}

        # Check if explicit split ("train" / "val" / "test") is provided (e.g. from keypoints_splited)
        has_explicit_split = any(r.get("split") in ("train", "val", "test") for r in rows)
        if has_explicit_split:
            train_x, train_y = [], []
            val_x, val_y = [], []
            test_x, test_y = [], []
            for r in rows:
                sp = str(r.get("split", "train")).lower()
                y_idx = cls_to_idx[normalize_single_gloss(r["label"])]
                if sp == "val":
                    val_x.append(r["sequence"])
                    val_y.append(y_idx)
                elif sp == "test":
                    test_x.append(r["sequence"])
                    test_y.append(y_idx)
                else:
                    train_x.append(r["sequence"])
                    train_y.append(y_idx)
            if not val_x:
                val_x, val_y = train_x, train_y
            if not test_x:
                test_x, test_y = val_x, val_y
        else:
            by_group: Dict[str, List[int]] = {}
            x_all = []
            y_all = []
            for idx, r in enumerate(rows):
                x_all.append(r["sequence"])
                y_all.append(cls_to_idx[normalize_single_gloss(r["label"])])
                gid = str(r.get("speakerId") or r.get("recordingId") or f"g_{idx}")
                by_group.setdefault(gid, []).append(idx)
            groups = sorted(by_group.keys())
            rng = random.Random(seed)
            rng.shuffle(groups)
            if len(groups) >= 3:
                n_tr = max(1, len(groups) - 2)
                tr_g = set(groups[:n_tr])
                va_g = {groups[n_tr]}
                te_g = set(groups[n_tr + 1 :])
            else:
                tr_g = set(groups)
                va_g = set(groups)
                te_g = set(groups)
            tr_idx = [i for g in tr_g for i in by_group[g]]
            va_idx = [i for g in va_g for i in by_group[g]]
            te_idx = [i for g in te_g for i in by_group[g]]
            train_x = [x_all[i] for i in tr_idx]
            train_y = [y_all[i] for i in tr_idx]
            val_x = [x_all[i] for i in va_idx]
            val_y = [y_all[i] for i in va_idx]
            test_x = [x_all[i] for i in te_idx]
            test_y = [y_all[i] for i in te_idx]
    else:
        split_res = split_single_sign_dataset(rows, seed=seed)
        class_names = split_res["label_map"]
        train_x, train_y = split_res["train_X"], split_res["train_Y"]
        val_x, val_y = split_res["val_X"], split_res["val_Y"]
        test_x, test_y = split_res["test_X"], split_res["test_Y"]
        if not val_x:
            val_x, val_y = train_x, train_y
        if not test_x:
            test_x, test_y = val_x, val_y

    train_ds = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(val_x, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(test_x, dtype=torch.float32),
        torch.tensor(test_y, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = VSLSingleSignBaselineModel(
        num_classes=len(class_names),
        input_dim=SINGLE_SIGN_FEATURE_DIM,
        sequence_length=SINGLE_SIGN_SEQUENCE_LENGTH,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=max(1e-4, lr * 0.08)
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model.forward_logits(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            train_loss_sum += float(loss.item())
            batches += 1

        scheduler.step()

        val_metrics = evaluate_single_sign_model(
            model=model,
            dataloader=val_loader,
            class_names=class_names,
            device=device,
        )
        val_acc = float(val_metrics["accuracy"])
        val_loss = float(val_metrics["loss"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / float(max(1, batches)),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )

        if val_acc > best_val_acc or (
            abs(val_acc - best_val_acc) <= 1e-6 and val_loss < best_val_loss
        ):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": model.get_architecture_config(),
                    "feature_dim": SINGLE_SIGN_FEATURE_DIM,
                    "sequence_length": SINGLE_SIGN_SEQUENCE_LENGTH,
                    "class_names": class_names,
                    "num_classes": len(class_names),
                    "epoch": epoch,
                    "best_val_accuracy": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "seed": seed,
                },
                output_model_path,
            )

    best_ckpt = torch.load(output_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate_single_sign_model(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        device=device,
    )

    return {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "checkpoint_path": output_model_path,
        "class_names": class_names,
        "history": history,
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train VSL Continuous Recognition Model (CTC) or Single-Sign Baseline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset file (JSON or CSV).",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="checkpoints/vsl_ctc_model.pth",
        help="Path to save best model checkpoint (.pth).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ctc", "single_sign"],
        default="ctc",
        help="Training mode: 'ctc' (continuous sequence) or 'single_sign' (isolated 30x138).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    train_cfg = cfg.get("training", {})

    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 80))
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(train_cfg.get("batch_size", 16))
    )
    lr = args.lr if args.lr is not None else float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(train_cfg.get("weight_decay", 1e-4))
    )
    grad_clip = (
        args.grad_clip
        if args.grad_clip is not None
        else float(train_cfg.get("grad_clip_norm", 2.0))
    )
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else int(train_cfg.get("early_stopping_patience", 15))
    )

    if args.mode == "ctc":
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        raw_samples = raw_data if isinstance(raw_data, list) else raw_data.get("samples", [])
        result = train_ctc_pipeline(
            raw_samples=raw_samples,
            output_model_path=args.output_model,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip,
            seed=seed,
            early_stopping_patience=patience,
            device_str=args.device,
        )
    else:
        rows = load_single_sign_rows(args.dataset_path)
        result = train_single_sign_pipeline(
            rows=rows,
            output_model_path=args.output_model,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            device_str=args.device,
        )

    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "history"},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
