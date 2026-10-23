"""
Evaluation Pipeline for VSL Continuous Recognition Model & Single-Sign Baseline.

Provides:
1. Continuous CTC model evaluation (`evaluate_ctc_model`):
   - Average CTC loss
   - Corpus-level and sample-averaged Gloss Word Error Rate (WER)
   - Token-level Levenshtein breakdown (Substitutions S, Deletions D, Insertions I, Hits H)
   - Character Error Rate (CER)
   - Sequence Exact-Match Rate
   - Per-speaker performance breakdown
   - Sequence length bucket breakdown (frame count)
   - Gloss count bucket breakdown (sentence length in glosses)
   - Inference latency per sequence (ms) and per frame (ms)
2. Single-Sign Classification baseline evaluation (`evaluate_single_sign_model`):
   - Top-1 and Top-3 Accuracy
   - Macro F1 and Per-Class Precision / Recall / F1 / Support
   - Confusion Matrix
   - Rejection behavior for UNKNOWN (max_prob < 0.55) and AMBIGUOUS (margin < 0.12)
3. Boundary Detection evaluation (`evaluate_boundary_detector_on_raw_samples`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    VSLContinuousDataset,
    load_continuous_json_dataset,
    load_single_sign_rows,
    pad_collate_fn,
    split_dataset_by_group,
    split_single_sign_dataset,
)
from decoder import (
    Vocabulary,
    compute_cer,
    compute_edit_distance_detail,
    compute_sequence_exact_match,
    decode_ctc_beam_search,
    decode_ctc_greedy_detailed,
)
from feature_extractor import (
    CTC_FEATURE_DIM,
    FEATURE_VERSION,
    SINGLE_SIGN_FEATURE_DIM,
    SignBoundaryDetector,
    create_hand_track_state,
    extract_canonical_212_frame,
)
from model import VSLSignCTCModel, VSLSingleSignBaselineModel


def _collate_with_meta(batch: Sequence[Dict[str, Any]]) -> Tuple[ Any, ... ]:
    padded_features, concatenated_targets, input_lengths, target_lengths, glosses = pad_collate_fn(batch)
    meta_list = [
        {
            "id": item.get("id", ""),
            "speaker_id": item.get("speaker_id", "unknown"),
            "session_id": item.get("session_id", "unknown"),
            "recording_id": item.get("recording_id", "unknown"),
        }
        for item in batch
    ]
    return padded_features, concatenated_targets, input_lengths, target_lengths, glosses, meta_list


def evaluate_ctc_model(
    model: VSLSignCTCModel,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    vocab: Vocabulary,
    device: torch.device,
    decode_method: str = "greedy",
    beam_width: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate a trained `VSLSignCTCModel` on a validation or test DataLoader.
    Never mutates model parameters or BatchNorm running statistics.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_hits = 0
    total_ref_tokens = 0
    total_exact_matches = 0
    total_samples = 0
    total_frames = 0
    total_inference_sec = 0.0

    sample_wers: List[float] = []
    sample_cers: List[float] = []
    per_speaker_stats: Dict[str, Dict[str, Any]] = {}
    length_bucket_stats: Dict[str, Dict[str, Any]] = {
        "short_lt_40f": {"samples": 0, "edits": 0, "ref_tokens": 0, "exact_matches": 0},
        "medium_40_to_80f": {"samples": 0, "edits": 0, "ref_tokens": 0, "exact_matches": 0},
        "long_gt_80f": {"samples": 0, "edits": 0, "ref_tokens": 0, "exact_matches": 0},
    }
    gloss_count_bucket_stats: Dict[str, Dict[str, Any]] = {
        "single_gloss_1": {"samples": 0, "edits": 0, "ref_tokens": 0, "exact_matches": 0},
        "short_phrase_2_3": {"samples": 0, "edits": 0, "ref_tokens": 0, "exact_matches": 0},
        "long_sentence_4_plus": {"samples": 0, "edits": 0, "ref_tokens": 0, "exact_matches": 0},
    }
    sample_predictions: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 6:
                features, targets, input_lengths, target_lengths, batch_glosses, meta_list = batch
            else:
                features, targets, input_lengths, target_lengths, batch_glosses = batch
                meta_list = [{} for _ in range(features.size(0))]

            features = features.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            t0 = time.perf_counter()
            log_probs = model(features, input_lengths)  # [B, T, C]
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_sec = time.perf_counter() - t0
            total_inference_sec += elapsed_sec

            log_probs_tbc = log_probs.transpose(0, 1)  # [T, B, C]
            loss = criterion(log_probs_tbc, targets, input_lengths, target_lengths)
            if math.isfinite(loss.item()):
                total_loss += float(loss.item())
                total_batches += 1

            log_probs_np = log_probs.detach().cpu().numpy()
            input_lengths_list = input_lengths.detach().cpu().tolist()
            target_lengths_list = target_lengths.detach().cpu().tolist()
            targets_list = targets.detach().cpu().tolist()

            target_offset = 0
            for i, seq_len in enumerate(input_lengths_list):
                t_len = target_lengths_list[i]
                ref_ids = targets_list[target_offset : target_offset + t_len]
                target_offset += t_len

                ref_glosses = batch_glosses[i] if i < len(batch_glosses) else vocab.decode(ref_ids)
                valid_log_probs = log_probs_np[i, :seq_len, :]

                if decode_method == "beam_search":
                    hyp_glosses, hyp_ids = decode_ctc_beam_search(
                        valid_log_probs, vocab=vocab, beam_width=beam_width, is_log_probs=True
                    )
                else:
                    greedy_res = decode_ctc_greedy_detailed(valid_log_probs, vocab=vocab)
                    hyp_glosses = greedy_res.glosses
                    hyp_ids = greedy_res.decoded_token_ids

                edit_detail = compute_edit_distance_detail(ref_glosses, hyp_glosses)
                cer_val = compute_cer(ref_glosses, hyp_glosses)
                exact_match = compute_sequence_exact_match(ref_glosses, hyp_glosses)

                s = int(edit_detail.substitutions)
                d = int(edit_detail.deletions)
                ins = int(edit_detail.insertions)
                h = int(edit_detail.hits)
                edits = int(edit_detail.edit_distance)
                n_ref = len(ref_glosses)

                total_substitutions += s
                total_deletions += d
                total_insertions += ins
                total_hits += h
                total_ref_tokens += n_ref
                total_exact_matches += 1 if exact_match else 0
                total_samples += 1
                total_frames += int(seq_len)

                sample_wers.append(float(edit_detail.wer))
                sample_cers.append(float(cer_val))

                meta = meta_list[i] if i < len(meta_list) else {}
                speaker_id = str(meta.get("speaker_id", "unknown"))
                if speaker_id not in per_speaker_stats:
                    per_speaker_stats[speaker_id] = {
                        "samples": 0,
                        "edits": 0,
                        "ref_tokens": 0,
                        "exact_matches": 0,
                        "substitutions": 0,
                        "deletions": 0,
                        "insertions": 0,
                        "hits": 0,
                    }
                sp = per_speaker_stats[speaker_id]
                sp["samples"] += 1
                sp["edits"] += edits
                sp["ref_tokens"] += n_ref
                sp["exact_matches"] += 1 if exact_match else 0
                sp["substitutions"] += s
                sp["deletions"] += d
                sp["insertions"] += ins
                sp["hits"] += h

                if seq_len < 40:
                    lb_key = "short_lt_40f"
                elif seq_len <= 80:
                    lb_key = "medium_40_to_80f"
                else:
                    lb_key = "long_gt_80f"
                length_bucket_stats[lb_key]["samples"] += 1
                length_bucket_stats[lb_key]["edits"] += edits
                length_bucket_stats[lb_key]["ref_tokens"] += n_ref
                length_bucket_stats[lb_key]["exact_matches"] += 1 if exact_match else 0

                if n_ref <= 1:
                    gb_key = "single_gloss_1"
                elif n_ref <= 3:
                    gb_key = "short_phrase_2_3"
                else:
                    gb_key = "long_sentence_4_plus"
                gloss_count_bucket_stats[gb_key]["samples"] += 1
                gloss_count_bucket_stats[gb_key]["edits"] += edits
                gloss_count_bucket_stats[gb_key]["ref_tokens"] += n_ref
                gloss_count_bucket_stats[gb_key]["exact_matches"] += 1 if exact_match else 0

                sample_predictions.append(
                    {
                        "id": meta.get("id", f"sample_{total_samples - 1}"),
                        "speaker_id": speaker_id,
                        "frames": int(seq_len),
                        "reference": ref_glosses,
                        "hypothesis": hyp_glosses,
                        "reference_ids": ref_ids,
                        "hypothesis_ids": hyp_ids,
                        "wer": float(edit_detail.wer),
                        "cer": float(cer_val),
                        "exact_match": bool(exact_match),
                        "substitutions": s,
                        "deletions": d,
                        "insertions": ins,
                        "hits": h,
                    }
                )

    avg_loss = total_loss / max(1, total_batches)
    total_edits = total_substitutions + total_deletions + total_insertions
    corpus_wer = (
        float(total_edits) / float(total_ref_tokens)
        if total_ref_tokens > 0
        else (0.0 if total_edits == 0 else 1.0)
    )
    mean_sample_wer = float(np.mean(sample_wers)) if sample_wers else 0.0
    mean_sample_cer = float(np.mean(sample_cers)) if sample_cers else 0.0
    exact_match_rate = float(total_exact_matches) / float(max(1, total_samples))

    speaker_summary: Dict[str, Dict[str, Any]] = {}
    for spk, st in per_speaker_stats.items():
        n_samp = max(1, st["samples"])
        n_tok = st["ref_tokens"]
        speaker_summary[spk] = {
            **st,
            "wer": float(st["edits"]) / float(n_tok) if n_tok > 0 else 0.0,
            "exact_match_rate": float(st["exact_matches"]) / float(n_samp),
        }

    def _finalize_bucket(bucket_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k, st in bucket_dict.items():
            n_samp = st["samples"]
            n_tok = st["ref_tokens"]
            out[k] = {
                **st,
                "wer": float(st["edits"]) / float(n_tok) if n_tok > 0 else 0.0,
                "exact_match_rate": float(st["exact_matches"]) / float(n_samp) if n_samp > 0 else 0.0,
            }
        return out

    latency_ms_per_sequence = (total_inference_sec * 1000.0) / float(max(1, total_samples))
    latency_ms_per_frame = (total_inference_sec * 1000.0) / float(max(1, total_frames))

    return {
        "loss": avg_loss,
        "corpus_wer": corpus_wer,
        "mean_sample_wer": mean_sample_wer,
        "mean_sample_cer": mean_sample_cer,
        "exact_match_rate": exact_match_rate,
        "total_samples": total_samples,
        "total_frames": total_frames,
        "total_ref_tokens": total_ref_tokens,
        "substitutions": total_substitutions,
        "deletions": total_deletions,
        "insertions": total_insertions,
        "hits": total_hits,
        "total_edits": total_edits,
        "decode_method": decode_method,
        "latency_ms_per_sequence": latency_ms_per_sequence,
        "latency_ms_per_frame": latency_ms_per_frame,
        "per_speaker_metrics": speaker_summary,
        "length_bucket_metrics": _finalize_bucket(length_bucket_stats),
        "gloss_count_bucket_metrics": _finalize_bucket(gloss_count_bucket_stats),
        "predictions": sample_predictions,
    }


def evaluate_single_sign_model(
    model: VSLSingleSignBaselineModel,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
    unknown_threshold: float = 0.55,
    ambiguity_margin: float = 0.12,
) -> Dict[str, Any]:
    """
    Evaluate `VSLSingleSignBaselineModel` on isolated 30x138 sign sequences.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    num_classes = len(class_names)

    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    top1_correct = 0
    top3_correct = 0

    unknown_rejected = 0
    ambiguous_rejected = 0
    accepted_samples = 0
    accepted_correct = 0

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_inference_sec = 0.0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            t0 = time.perf_counter()
            logits = model.forward_logits(features)  # [B, num_classes]
            probs_tensor = torch.softmax(logits, dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_inference_sec += time.perf_counter() - t0

            loss = criterion(logits, labels)
            if math.isfinite(loss.item()):
                total_loss += float(loss.item())
                total_batches += 1

            probs = probs_tensor.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(labels_np)):
                true_idx = int(labels_np[i])
                prob_vec = probs[i]
                sorted_indices = np.argsort(prob_vec)[::-1]
                pred_idx = int(sorted_indices[0])
                top1_prob = float(prob_vec[pred_idx])
                top2_prob = float(prob_vec[sorted_indices[1]]) if num_classes > 1 else 0.0

                total_samples += 1
                if pred_idx == true_idx:
                    top1_correct += 1
                if true_idx in sorted_indices[: min(3, num_classes)]:
                    top3_correct += 1

                if 0 <= true_idx < num_classes and 0 <= pred_idx < num_classes:
                    confusion_matrix[true_idx, pred_idx] += 1

                if top1_prob < unknown_threshold:
                    unknown_rejected += 1
                elif num_classes > 1 and (top1_prob - top2_prob) < ambiguity_margin:
                    ambiguous_rejected += 1
                else:
                    accepted_samples += 1
                    if pred_idx == true_idx:
                        accepted_correct += 1

    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    macro_f1_list: List[float] = []
    weighted_f1_sum = 0.0

    for c_idx, c_name in enumerate(class_names):
        tp = int(confusion_matrix[c_idx, c_idx])
        fp = int(confusion_matrix[:, c_idx].sum() - tp)
        fn = int(confusion_matrix[c_idx, :].sum() - tp)
        support = int(confusion_matrix[c_idx, :].sum())

        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

        per_class_metrics[c_name] = {
            "class_id": c_idx,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support > 0:
            macro_f1_list.append(f1)
            weighted_f1_sum += f1 * support

    accuracy = float(top1_correct) / float(max(1, total_samples))
    top3_accuracy = float(top3_correct) / float(max(1, total_samples))
    macro_f1 = float(np.mean(macro_f1_list)) if macro_f1_list else 0.0
    weighted_f1 = weighted_f1_sum / float(max(1, total_samples))
    accepted_accuracy = (
        float(accepted_correct) / float(accepted_samples) if accepted_samples > 0 else 0.0
    )

    return {
        "loss": total_loss / float(max(1, total_batches)),
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_samples": total_samples,
        "rejection_analysis": {
            "unknown_threshold": unknown_threshold,
            "ambiguity_margin": ambiguity_margin,
            "unknown_rejected": unknown_rejected,
            "ambiguous_rejected": ambiguous_rejected,
            "accepted_samples": accepted_samples,
            "coverage_rate": float(accepted_samples) / float(max(1, total_samples)),
            "accepted_accuracy": accepted_accuracy,
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": class_names,
        "latency_ms_per_sequence": (total_inference_sec * 1000.0) / float(max(1, total_samples)),
    }


def evaluate_boundary_detector_on_raw_samples(
    raw_samples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate `SignBoundaryDetector` segmentation behavior on raw landmark sequences.
    """
    detector = SignBoundaryDetector()
    total_recordings = 0
    total_detected_segments = 0
    recording_details: List[Dict[str, Any]] = []

    for idx, sample in enumerate(raw_samples):
        frames = sample.get("frames")
        if not isinstance(frames, list) or len(frames) == 0:
            continue

        detector.reset()
        tracker = create_hand_track_state()
        total_recordings += 1
        segments_in_recording = 0

        for f_idx, frame in enumerate(frames):
            ts = float(frame.get("timestamp", f_idx * 33.333))
            hands = frame.get("rawHands") or frame.get("hands")
            if not isinstance(hands, list) or (
                len(hands) > 0 and isinstance(hands[0], dict) and "landmarks" in hands[0]
            ):
                if isinstance(hands, list) and len(hands) > 0 and isinstance(hands[0], dict):
                    hands_lms = [h["landmarks"] for h in hands if "landmarks" in h]
                    h_labels = [h.get("handedness") for h in hands if "landmarks" in h]
                else:
                    lm = frame.get("landmarks")
                    hands_lms = [lm] if isinstance(lm, list) and len(lm) >= 21 else []
                    h_labels = frame.get("handedness")
            else:
                hands_lms = hands
                h_labels = frame.get("handedness")

            res212 = extract_canonical_212_frame(hands_lms, h_labels, tracker)
            seg = detector.process_frame(
                timestamp=ts,
                feature_vector=res212.combined212,
                hands_landmarks=hands_lms,
                hand_labels=h_labels,
            )
            if seg is not None:
                segments_in_recording += 1
                total_detected_segments += 1

        recording_details.append(
            {
                "id": sample.get("sentenceId") or sample.get("id", f"sample_{idx}"),
                "num_frames": len(frames),
                "expected_gloss_count": len(sample.get("glossSequence") or sample.get("glosses") or []),
                "detected_segments": segments_in_recording,
            }
        )

    return {
        "total_raw_recordings": total_recordings,
        "total_detected_segments": total_detected_segments,
        "recordings": recording_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VSL Continuous CTC Model or Single-Sign Baseline."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PyTorch checkpoint (.pth).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to evaluation dataset file (JSON or CSV).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ctc", "single_sign"],
        default="ctc",
        help="Evaluation mode: 'ctc' (continuous sequence) or 'single_sign' (isolated 30x138).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "all"],
        default="test",
        help="Which split to evaluate when group splitting is applied (default: test).",
    )
    parser.add_argument(
        "--decode_method",
        type=str,
        choices=["greedy", "beam_search"],
        default="greedy",
        help="CTC decoding strategy: 'greedy' or 'beam_search'.",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=10,
        help="Beam width when --decode_method=beam_search.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device ('cpu', 'cuda', or 'auto').",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="",
        help="Optional path to save JSON evaluation report.",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.mode == "ctc":
        vocab_data = ckpt.get("vocabulary") or ckpt.get("id_to_gloss")
        vocab = Vocabulary.from_checkpoint_vocab(vocab_data) if vocab_data else Vocabulary()

        valid_samples, skipped, diag, _ = load_continuous_json_dataset(
            args.dataset_path, vocab=vocab
        )
        if args.split == "all":
            eval_samples = valid_samples
        else:
            _, val_samples, test_samples = split_dataset_by_group(
                valid_samples, seed=int(ckpt.get("seed", 42))
            )
            eval_samples = val_samples if args.split == "val" else test_samples

        dataset = VSLContinuousDataset(eval_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate_with_meta,
        )

        arch = ckpt.get("architecture", {})
        model = VSLSignCTCModel(
            input_dim=int(ckpt.get("feature_dim", CTC_FEATURE_DIM)),
            hidden_dim=int(arch.get("hidden_dim", 128)),
            num_classes=int(ckpt.get("num_classes", vocab.num_classes)),
            conv_channels=int(arch.get("conv_channels", 64)),
            lstm_layers=int(arch.get("lstm_layers", 2)),
            fc_hidden_dim=int(arch.get("fc_hidden_dim", 64)),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        metrics = evaluate_ctc_model(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            vocab=vocab,
            device=device,
            decode_method=args.decode_method,
            beam_width=args.beam_width,
        )
        metrics["skipped_samples"] = skipped
    else:
        rows = load_single_sign_rows(args.dataset_path)
        split_res = split_single_sign_dataset(rows, seed=int(ckpt.get("seed", 42)))
        class_names = ckpt.get("class_names") or split_res["label_map"]
        if args.split == "val":
            x_eval, y_eval = split_res["val_X"], split_res["val_Y"]
        elif args.split == "test":
            x_eval, y_eval = split_res["test_X"], split_res["test_Y"]
        else:
            x_eval = split_res["train_X"] + split_res["val_X"] + split_res["test_X"]
            y_eval = split_res["train_Y"] + split_res["val_Y"] + split_res["test_Y"]

        ds = TensorDataset(
            torch.tensor(x_eval, dtype=torch.float32),
            torch.tensor(y_eval, dtype=torch.long),
        )
        dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        model = VSLSingleSignBaselineModel(
            num_classes=len(class_names),
            input_dim=int(ckpt.get("feature_dim", SINGLE_SIGN_FEATURE_DIM)),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        metrics = evaluate_single_sign_model(
            model=model,
            dataloader=dataloader,
            class_names=class_names,
            device=device,
        )

    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k != "predictions"},
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.output_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluation report to {args.output_report}")


if __name__ == "__main__":
    main()
