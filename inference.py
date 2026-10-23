"""
Standalone Inference & Export Interface for VSL Continuous Recognition Model.

Provides:
1. `VSLContinuousRecognizer`:
   - Loads a saved `.pth` checkpoint and validates:
     * `feature_dim == 212`
     * `feature_version == "v1-canonical-left106-right106"`
     * `blank_id == 0` and `blank_token == "<blank>"`
     * Vocabulary mapping consistency
   - Accepts either:
     * Pre-extracted `[T, 212]` feature sequences (`predict_from_features`)
     * Raw landmark frame sequences (`predict_from_landmarks`, performing <=500ms occlusion
       interpolation and canonical `[Left(106), Right(106)]` feature extraction)
   - Supports both greedy CTC decoding and prefix beam search decoding.
   - Returns predicted glosses, token IDs, frame-level probabilities, sequence length,
     inference latency (ms), and model metadata.
2. `CTCStreamInferenceEngine`:
   - Exact Python port of `src/utils/ctcStreamInference.ts` rolling-window streaming
     engine (`window_size_frames=75`, `stride_frames=12`, `min_frames_to_infer=20`,
     consecutive-window stabilization).
3. `VSLSingleSignRecognizer`:
   - Standalone inference wrapper for the 30x138 single-sign baseline with
     `UNKNOWN` (`max_prob < 0.55`) and `AMBIGUOUS` (`top1 - top2 < 0.12`) rejection gates.
4. `export_model_to_torchscript`:
   - Exports a trained `VSLSignCTCModel` checkpoint to a traced TorchScript `.pt` bundle
     alongside its `.meta.json` contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from dataset import _extract_frame_212_vector
from decoder import (
    BLANK_ID,
    BLANK_TOKEN,
    Vocabulary,
    decode_ctc_beam_search,
    decode_ctc_greedy_detailed,
)
from feature_extractor import (
    COMPATIBLE_FEATURE_VERSIONS,
    CTC_FEATURE_DIM,
    FEATURE_VERSION,
    SINGLE_SIGN_FEATURE_DIM,
    SINGLE_SIGN_SEQUENCE_LENGTH,
    create_hand_track_state,
    extract_canonical_212_frame,
    interpolate_occluded_frames,
    resample_sequence,
)
from model import VSLSignCTCModel, VSLSingleSignBaselineModel


def extract_212_sequence_from_raw_frames(
    raw_frames: Sequence[Dict[str, Any]],
    interpolate_gaps: bool = True,
    max_gap_ms: float = 500.0,
) -> List[List[float]]:
    """
    Convert a sequence of raw frame dicts into a sequence of canonical 212-D vectors
    (`Left 106 + Right 106`), interpolating short internal occlusion gaps <= 500ms.
    """
    tracker = create_hand_track_state()
    enriched_frames: List[Dict[str, Any]] = []

    for idx, f in enumerate(raw_frames):
        f_copy = dict(f)
        if "timestamp" not in f_copy:
            f_copy["timestamp"] = idx * 33.333

        vec = f_copy.get("combinedFeatureVector")
        if isinstance(vec, (list, tuple)) and len(vec) == CTC_FEATURE_DIM:
            f_copy["combinedFeatureVector"] = list(vec)
            f_copy["isOccluded"] = bool(f_copy.get("isOccluded", all(abs(x) < 1e-8 for x in vec)))
            f_copy["handCount"] = int(f_copy.get("handCount", 0 if f_copy["isOccluded"] else 1))
            enriched_frames.append(f_copy)
            continue

        vec_212 = _extract_frame_212_vector(f_copy, is_explicit_canonical=True, tracker=tracker)
        if vec_212 is None or all(abs(x) < 1e-8 for x in vec_212):
            f_copy["combinedFeatureVector"] = [0.0] * CTC_FEATURE_DIM
            f_copy["isOccluded"] = True
            f_copy["handCount"] = 0
        else:
            f_copy["combinedFeatureVector"] = vec_212
            f_copy["isOccluded"] = False
            f_copy["handCount"] = 1
        enriched_frames.append(f_copy)

    if interpolate_gaps and len(enriched_frames) > 2:
        enriched_frames, _, _ = interpolate_occluded_frames(enriched_frames, max_gap_ms=max_gap_ms)

    return [f["combinedFeatureVector"] for f in enriched_frames]


class VSLContinuousRecognizer:
    """
    Standalone inference engine for the VSL Continuous Recognition Model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        strict_contract_check: bool = True,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self._validate_checkpoint_contract(ckpt, strict=strict_contract_check)

        self.feature_dim = int(ckpt.get("feature_dim", CTC_FEATURE_DIM))
        self.feature_version = str(ckpt.get("feature_version", FEATURE_VERSION))
        self.blank_id = int(ckpt.get("blank_id", BLANK_ID))
        self.blank_token = str(ckpt.get("blank_token", BLANK_TOKEN))

        vocab_data = ckpt.get("vocabulary") or ckpt.get("id_to_gloss")
        if vocab_data is not None:
            self.vocab = Vocabulary.from_checkpoint_vocab(vocab_data)
        else:
            self.vocab = Vocabulary()

        arch = ckpt.get("architecture", {})
        self.model = VSLSignCTCModel(
            input_dim=self.feature_dim,
            hidden_dim=int(arch.get("hidden_dim", 128)),
            num_classes=int(ckpt.get("num_classes", self.vocab.num_classes)),
            conv_channels=int(arch.get("conv_channels", 64)),
            lstm_layers=int(arch.get("lstm_layers", 2)),
            fc_hidden_dim=int(arch.get("fc_hidden_dim", 64)),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.metadata: Dict[str, Any] = {
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "feature_dim": self.feature_dim,
            "feature_version": self.feature_version,
            "blank_id": self.blank_id,
            "blank_token": self.blank_token,
            "num_classes": self.vocab.num_classes,
            "vocabulary": [self.vocab.id_to_word[i] for i in range(self.vocab.num_classes)],
            "epoch": ckpt.get("epoch"),
            "best_val_loss": ckpt.get("best_val_loss"),
            "best_val_wer": ckpt.get("best_val_wer"),
            "seed": ckpt.get("seed"),
        }

    @staticmethod
    def _validate_checkpoint_contract(ckpt: Dict[str, Any], strict: bool = True) -> None:
        if "model_state_dict" not in ckpt:
            raise ValueError("Checkpoint is missing 'model_state_dict'.")

        feat_dim = int(ckpt.get("feature_dim", CTC_FEATURE_DIM))
        if feat_dim != CTC_FEATURE_DIM:
            raise ValueError(
                f"Feature dimension mismatch: expected {CTC_FEATURE_DIM}, got {feat_dim}"
            )

        feat_ver = str(ckpt.get("feature_version", FEATURE_VERSION))
        if strict and feat_ver not in COMPATIBLE_FEATURE_VERSIONS:
            raise ValueError(
                f"Feature version mismatch: expected '{FEATURE_VERSION}', got '{feat_ver}'"
            )

        blank_id = int(ckpt.get("blank_id", BLANK_ID))
        if blank_id != BLANK_ID:
            raise ValueError(f"CTC blank_id must be {BLANK_ID}, got {blank_id}")

        blank_token = str(ckpt.get("blank_token", BLANK_TOKEN))
        if strict and blank_token != BLANK_TOKEN:
            raise ValueError(f"CTC blank_token must be '{BLANK_TOKEN}', got '{blank_token}'")

    def predict_from_features(
        self,
        features: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
        decode_method: str = "greedy",
        beam_width: int = 10,
    ) -> Dict[str, Any]:
        """
        Run inference on a pre-extracted 212-dimensional feature sequence `[T, 212]`.
        """
        if isinstance(features, torch.Tensor):
            feat_np = features.detach().cpu().numpy().astype(np.float32)
        else:
            feat_np = np.asarray(features, dtype=np.float32)

        if feat_np.ndim == 3 and feat_np.shape[0] == 1:
            feat_np = feat_np[0]

        if feat_np.ndim != 2 or feat_np.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected 2D feature sequence of shape [T, {self.feature_dim}], "
                f"got {feat_np.shape}"
            )
        if feat_np.shape[0] == 0:
            raise ValueError("Input feature sequence is empty (T=0).")

        if not np.all(np.isfinite(feat_np)):
            feat_np = np.nan_to_num(feat_np, nan=0.0, posinf=0.0, neginf=0.0)

        seq_len = int(feat_np.shape[0])
        x_tensor = torch.from_numpy(feat_np).unsqueeze(0).to(self.device)  # [1, T, 212]
        lengths_tensor = torch.tensor([seq_len], dtype=torch.long, device=self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            log_probs_tensor = self.model(x_tensor, lengths_tensor)  # [1, T, C]
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        log_probs = log_probs_tensor[0, :seq_len, :].detach().cpu().numpy()
        probs = np.exp(log_probs)

        greedy_detail = decode_ctc_greedy_detailed(log_probs, vocab=self.vocab)
        if decode_method == "beam_search":
            beam_glosses, beam_ids = decode_ctc_beam_search(
                log_probs, vocab=self.vocab, beam_width=beam_width, is_log_probs=True
            )
            predicted_glosses = beam_glosses
            predicted_ids = beam_ids
        else:
            predicted_glosses = greedy_detail.glosses
            predicted_ids = greedy_detail.decoded_token_ids

        return {
            "predicted_glosses": predicted_glosses,
            "predicted_token_ids": predicted_ids,
            "raw_frame_predictions": greedy_detail.raw_token_ids,
            "collapsed_token_ids": greedy_detail.collapsed_token_ids,
            "frame_probabilities": probs,
            "frame_log_probabilities": log_probs,
            "sequence_length": seq_len,
            "decode_method": decode_method,
            "latency_ms": float(latency_ms),
            "model_metadata": self.metadata,
        }

    def predict_from_landmarks(
        self,
        raw_frames: Sequence[Dict[str, Any]],
        decode_method: str = "greedy",
        beam_width: int = 10,
    ) -> Dict[str, Any]:
        """
        Accept raw landmark frames (`hands`/`rawHands`/`landmarks`, `timestamp`, `handedness`),
        perform <=500ms occlusion interpolation, extract canonical 212-D features,
        and run CTC sequence recognition.
        """
        feat_seq = extract_212_sequence_from_raw_frames(raw_frames)
        if len(feat_seq) == 0:
            raise ValueError(
                "No valid hand landmark frames were found in the supplied raw_frames."
            )
        return self.predict_from_features(
            feat_seq, decode_method=decode_method, beam_width=beam_width
        )


class CTCStreamInferenceEngine:
    """
    Python port of `src/utils/ctcStreamInference.ts`.

    Maintains a rolling buffer of 212-D canonical frames (`window_size_frames=75`,
    `stride_frames=12`, `min_frames_to_infer=20`) and emits stabilized CTC gloss
    sequences during continuous signing.
    """

    def __init__(
        self,
        recognizer: VSLContinuousRecognizer,
        window_size_frames: int = 75,
        stride_frames: int = 12,
        min_frames_to_infer: int = 20,
        decode_method: str = "greedy",
        beam_width: int = 10,
    ) -> None:
        self.recognizer = recognizer
        self.window_size_frames = window_size_frames
        self.stride_frames = stride_frames
        self.min_frames_to_infer = min_frames_to_infer
        self.decode_method = decode_method
        self.beam_width = beam_width

        self.frame_buffer: List[List[float]] = []
        self.tracker = create_hand_track_state()
        self.frames_since_last_infer: int = 0
        self.last_decoded_glosses: List[str] = []
        self.stabilized_glosses: List[str] = []

    def reset(self) -> None:
        self.frame_buffer = []
        self.tracker = create_hand_track_state()
        self.frames_since_last_infer = 0
        self.last_decoded_glosses = []
        self.stabilized_glosses = []

    def push_landmarks(
        self,
        hands: Sequence[Sequence[Any]],
        handedness_labels: Optional[Sequence[Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract a 212-D canonical frame from live hands and push it to the rolling window.
        """
        if not hands or len(hands) == 0:
            self.tracker = create_hand_track_state()
            return None

        res = extract_canonical_212_frame(hands, handedness_labels, self.tracker)
        return self.push_feature_frame(res.combined212)

    def push_feature_frame(self, frame_212: Sequence[float]) -> Optional[Dict[str, Any]]:
        """
        Push a pre-extracted 212-D feature frame and run inference whenever
        `len(frame_buffer) >= min_frames_to_infer` and `frames_since_last_infer >= stride_frames`.
        """
        if len(frame_212) != CTC_FEATURE_DIM:
            raise ValueError(
                f"Expected {CTC_FEATURE_DIM}-D frame, got {len(frame_212)}"
            )
        self.frame_buffer.append(list(frame_212))
        if len(self.frame_buffer) > self.window_size_frames:
            self.frame_buffer = self.frame_buffer[-self.window_size_frames :]

        self.frames_since_last_infer += 1
        if (
            len(self.frame_buffer) < self.min_frames_to_infer
            or self.frames_since_last_infer < self.stride_frames
        ):
            return None

        self.frames_since_last_infer = 0
        res = self.recognizer.predict_from_features(
            self.frame_buffer,
            decode_method=self.decode_method,
            beam_width=self.beam_width,
        )
        decoded_glosses = res["predicted_glosses"]
        is_stabilized = (
            len(decoded_glosses) > 0
            and decoded_glosses == self.last_decoded_glosses
        )
        self.last_decoded_glosses = list(decoded_glosses)
        if is_stabilized:
            self.stabilized_glosses = list(decoded_glosses)

        return {
            "glosses": decoded_glosses,
            "token_ids": res["predicted_token_ids"],
            "is_stabilized": is_stabilized,
            "stabilized_glosses": list(self.stabilized_glosses),
            "buffer_length": len(self.frame_buffer),
            "latency_ms": res["latency_ms"],
        }


class VSLSingleSignRecognizer:
    """
    Standalone inference wrapper for the secondary single-sign classification baseline
    (`VSLSingleSignBaselineModel`, 30 frames x 138 features).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        unknown_threshold: float = 0.35,
        ambiguity_margin: float = 0.05,
    ) -> None:
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.class_names: List[str] = list(ckpt.get("class_names", []))
        self.unknown_threshold = unknown_threshold
        self.ambiguity_margin = ambiguity_margin

        arch = ckpt.get("architecture", {}) if isinstance(ckpt.get("architecture"), dict) else {}
        self.model = VSLSingleSignBaselineModel(
            num_classes=int(ckpt.get("num_classes", max(2, len(self.class_names)))),
            input_dim=int(ckpt.get("feature_dim", SINGLE_SIGN_FEATURE_DIM)),
            sequence_length=int(ckpt.get("sequence_length", SINGLE_SIGN_SEQUENCE_LENGTH)),
            conv_filters=int(arch.get("conv_filters", 64)),
            lstm_units=int(arch.get("lstm_units", 64)),
            dense_units=int(arch.get("dense_units", 64)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(
        self, sequence_138: Union[np.ndarray, Sequence[Sequence[float]]]
    ) -> Dict[str, Any]:
        seq_list = (
            sequence_138.tolist()
            if isinstance(sequence_138, np.ndarray)
            else [list(row) for row in sequence_138]
        )
        resampled = resample_sequence(
            seq_list,
            target_len=SINGLE_SIGN_SEQUENCE_LENGTH,
            expected_dim=SINGLE_SIGN_FEATURE_DIM,
        )
        arr = np.asarray(resampled, dtype=np.float32)
        if arr.shape != (SINGLE_SIGN_SEQUENCE_LENGTH, SINGLE_SIGN_FEATURE_DIM):
            raise ValueError(f"Expected resampled shape (30, 138), got {arr.shape}")

        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            probs = self.model(x)[0].cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        sorted_idx = np.argsort(probs)[::-1]
        top1_idx = int(sorted_idx[0])
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[sorted_idx[1]]) if len(probs) > 1 else 0.0

        label = (
            self.class_names[top1_idx]
            if top1_idx < len(self.class_names)
            else f"CLASS_{top1_idx}"
        )
        status = "ACCEPTED"
        if top1_prob < self.unknown_threshold:
            status = "UNKNOWN"
            label = "UNKNOWN"
        elif len(probs) > 1 and (top1_prob - top2_prob) < self.ambiguity_margin:
            status = "AMBIGUOUS"
            label = "AMBIGUOUS"

        return {
            "predicted_label": label,
            "raw_top1_label": (
                self.class_names[top1_idx]
                if top1_idx < len(self.class_names)
                else f"CLASS_{top1_idx}"
            ),
            "confidence": top1_prob,
            "margin": top1_prob - top2_prob,
            "status": status,
            "probabilities": probs.tolist(),
            "latency_ms": float(latency_ms),
        }


def export_model_to_torchscript(
    checkpoint_path: str,
    output_pt_path: str,
    example_seq_len: int = 45,
) -> Dict[str, str]:
    """
    Export a trained `VSLSignCTCModel` `.pth` checkpoint to a standalone TorchScript `.pt`
    file and companion `.meta.json` contract.
    """
    recognizer = VSLContinuousRecognizer(checkpoint_path, device="cpu")
    model = recognizer.model
    model.eval()

    dummy_input = torch.zeros(1, example_seq_len, recognizer.feature_dim, dtype=torch.float32)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (dummy_input,))

    os.makedirs(os.path.dirname(os.path.abspath(output_pt_path)), exist_ok=True)
    traced_model.save(output_pt_path)

    meta_path = (
        output_pt_path[:-3] + ".meta.json"
        if output_pt_path.endswith(".pt")
        else output_pt_path + ".meta.json"
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(recognizer.metadata, f, indent=2, ensure_ascii=False)

    return {"torchscript_path": output_pt_path, "metadata_path": meta_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone inference with VSLContinuousRecognizer."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PyTorch checkpoint (.pth).",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to JSON file containing a sequence sample (or list of samples).",
    )
    parser.add_argument(
        "--decode_method",
        type=str,
        choices=["greedy", "beam_search"],
        default="greedy",
    )
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--export_torchscript", type=str, default="")
    args = parser.parse_args()

    recognizer = VSLContinuousRecognizer(args.checkpoint)

    if args.export_torchscript:
        export_info = export_model_to_torchscript(
            args.checkpoint, args.export_torchscript
        )
        print("Exported TorchScript model:", json.dumps(export_info, indent=2))

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "samples" in payload:
        sample = payload["samples"][0]
    elif isinstance(payload, list):
        sample = payload[0]
    else:
        sample = payload

    if isinstance(sample, dict) and "features" in sample:
        result = recognizer.predict_from_features(
            sample["features"],
            decode_method=args.decode_method,
            beam_width=args.beam_width,
        )
    elif isinstance(sample, dict) and "frames" in sample:
        result = recognizer.predict_from_landmarks(
            sample["frames"],
            decode_method=args.decode_method,
            beam_width=args.beam_width,
        )
    else:
        raise ValueError("Input JSON must contain 'features' or 'frames'.")

    printable = {
        k: v
        for k, v in result.items()
        if k not in ("frame_probabilities", "frame_log_probabilities")
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
