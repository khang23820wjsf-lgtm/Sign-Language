#!/usr/bin/env python3
"""
===============================================================================
DATASET LOADER, CTC LENGTH VALIDATION & LEAKAGE-FREE GROUP SPLITTING
Supports:
1. Continuous Sentence CTC Datasets (`vsl_continuous_ctc_dataset.json`):
   - Pre-extracted 212-D canonical frames (`combinedFeatureVector` or `hands[].featureVector`).
   - Raw 21-point hand landmark frames (`rawHands` / `landmarks`), automatically extracted
     via `extract_canonical_212_frame`.
   - Strict P0-1 metadata validation (`sentenceId`, `speakerId`, `sessionId`, `recordingId`).
   - Strict P0-7 vocabulary validation (no OOV gloss fabrication).
   - Strict P0-4 CTC minimum-length validation:
     `minimum_required_T = target_length + number_of_adjacent_repeated_target_pairs`.
   - Pre-augmentation base recording grouping (`strip_augmentation_suffix`) and 3-tier
     hierarchical group splitting (`speakerId` -> `sessionId` -> `recordingId`) with zero leakage.
2. Single-Sign Classification Baseline Datasets (CSV or JSON `DatasetRow` format):
   - 138-D feature extraction + idle boundary trimming + uniform resampling to 30 frames.
   - Pre-augmentation recording identity parsing so augmented variants never cross splits.
===============================================================================
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from decoder import Vocabulary, canonicalize_gloss_sequence, normalize_single_gloss
from feature_extractor import (
    COMPATIBLE_FEATURE_VERSIONS,
    CTC_FEATURE_DIM,
    SINGLE_HAND_FEATURE_DIM,
    SINGLE_SIGN_FEATURE_DIM,
    SINGLE_SIGN_SEQUENCE_LENGTH,
    Landmark3D,
    create_hand_track_state,
    extract_canonical_212_frame,
    extract_invariant_feature_vector,
    resample_sequence,
    sanitize_float,
    sanitize_vector,
    trim_sequence_idle_boundaries,
)


# =============================================================================
# 1. CTC CONTINUOUS DATASET VALIDATION & LOADING
# =============================================================================

_AUG_SUFFIX_RE = re.compile(
    r"([_-](?:aug|rot|scale|noise|speed|shift)[_-]?[a-zA-Z0-9.\-+_]*)$",
    re.IGNORECASE,
)


def strip_augmentation_suffix(raw_id: str) -> str:
    """
    Strip augmentation suffixes (e.g. `rec_01_aug_1`, `rec_01_rot_-5deg`) to recover
    the canonical base recording ID so augmented copies never leak across splits.
    """
    s = str(raw_id or "").strip()
    prev = None
    while s and s != prev:
        prev = s
        s = _AUG_SUFFIX_RE.sub("", s)
    return s or str(raw_id or "").strip()


def compute_ctc_minimum_required_length(target_ids: Sequence[int]) -> Tuple[int, int]:
    """
    Compute the exact CTC minimum frame length requirement:
      `minimum_required_T = target_length + number_of_adjacent_repeated_target_pairs`
    Returns `(minimum_required_T, adjacent_repeats)`.
    """
    u = len(target_ids)
    adjacent_repeats = sum(1 for i in range(u - 1) if target_ids[i] == target_ids[i + 1])
    return u + adjacent_repeats, adjacent_repeats


@dataclass
class DatasetValidationDiagnostic:
    total_samples: int = 0
    valid_samples: int = 0
    rejected_samples: int = 0
    missing_sentence_id: int = 0
    missing_speaker_id: int = 0
    missing_session_id: int = 0
    missing_recording_id: int = 0
    empty_gloss_or_frames: int = 0
    unknown_gloss: int = 0
    invalid_canonical_212: int = 0
    ctc_input_too_short: int = 0
    rejection_logs: List[str] = field(default_factory=list)


class VSLContinuousDataset(Dataset):
    """PyTorch Dataset wrapping validated continuous VSL samples."""

    def __init__(self, sample_items: List[Dict[str, Any]]) -> None:
        self.samples = sample_items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def _extract_frame_212_vector(
    frame: Dict[str, Any],
    is_explicit_canonical: bool,
    tracker: Any,
) -> Optional[List[float]]:
    """
    Extract and sanitize a 212-D canonical vector from a single frame dict.
    Supports:
      1. `combinedFeatureVector` (length 212) when `featureVersion` is canonical.
      2. `hands` list containing `{handedness, featureVector}` (106-D each) or `{handedness, landmarks}` (21 points).
      3. `rawHands` / `landmarks` containing 21-point 3D coordinates.
    """
    vec = frame.get("combinedFeatureVector")
    if is_explicit_canonical and isinstance(vec, (list, tuple)) and len(vec) == CTC_FEATURE_DIM:
        return sanitize_vector(vec, CTC_FEATURE_DIM)

    hands = frame.get("hands")
    if isinstance(hands, list) and len(hands) > 0:
        # Check if hands contain raw 21-point landmarks or pre-extracted 106-D featureVector
        if any("landmarks" in h and isinstance(h["landmarks"], list) for h in hands if isinstance(h, dict)):
            raw_lms = [h["landmarks"] for h in hands if isinstance(h, dict) and "landmarks" in h]
            raw_h = [h.get("handedness", "Unknown") for h in hands if isinstance(h, dict) and "landmarks" in h]
            res = extract_canonical_212_frame(raw_lms, raw_h, tracker)
            return sanitize_vector(res.combined212, CTC_FEATURE_DIM)

        left_vec = [0.0] * SINGLE_HAND_FEATURE_DIM
        right_vec = [0.0] * SINGLE_HAND_FEATURE_DIM
        has_valid_hand_vec = False
        for h in hands:
            if not isinstance(h, dict):
                continue
            hname = str(h.get("handedness", "")).strip()
            hvec = h.get("featureVector")
            if isinstance(hvec, (list, tuple)) and len(hvec) == SINGLE_HAND_FEATURE_DIM:
                clean_106 = sanitize_vector(hvec, SINGLE_HAND_FEATURE_DIM)
                if hname in ("Tay trái", "Left", "left", "LEFT"):
                    left_vec = clean_106
                    has_valid_hand_vec = True
                elif hname in ("Tay phải", "Right", "right", "RIGHT"):
                    right_vec = clean_106
                    has_valid_hand_vec = True
        if has_valid_hand_vec:
            return left_vec + right_vec

    raw_hands = frame.get("rawHands") or frame.get("screenHands") or frame.get("landmarks")
    if isinstance(raw_hands, list):
        if len(raw_hands) == 0:
            res = extract_canonical_212_frame([], [], tracker)
            return res.combined212
        # Single hand passed as 21 landmarks directly vs list of hands
        if len(raw_hands) == 21 and isinstance(raw_hands[0], (dict, list, tuple)) and not (
            isinstance(raw_hands[0], list) and len(raw_hands[0]) == 21
        ):
            raw_hands = [raw_hands]
        raw_handedness = frame.get("handedness") or frame.get("rawHandedness") or []
        res = extract_canonical_212_frame(raw_hands, raw_handedness, tracker)
        return sanitize_vector(res.combined212, CTC_FEATURE_DIM)

    return None


def parse_and_validate_samples(
    raw_samples: Sequence[Dict[str, Any]],
    vocab: Vocabulary,
    min_sequence_length: int = 3,
    return_diagnostics: bool = False,
) -> Union[Tuple[List[Dict[str, Any]], int], Tuple[List[Dict[str, Any]], int, DatasetValidationDiagnostic]]:
    """
    Validate all continuous sentence samples against strict P0 contracts:
    - Real metadata required (`sentenceId`, `speakerId`, `sessionId`, `recordingId`) - never fabricated.
    - All glosses must exist in `vocab` (normalized via Unicode NFC + uppercase + underscores).
    - Every frame must yield a finite 212-D canonical feature vector (`LEFT 106 + RIGHT 106`).
    - Input length T must satisfy `T >= max(min_sequence_length, target_length + adjacent_repeats)`.
    """
    valid_samples: List[Dict[str, Any]] = []
    skipped_count = 0
    diag = DatasetValidationDiagnostic(total_samples=len(raw_samples))

    for s in raw_samples:
        s_id_raw = s.get("sentenceId")
        if not s_id_raw or not str(s_id_raw).strip():
            msg = f"[CTC_DATASET] rejected sample {s.get('sentenceId', 'unknown')} reason=missing_sentence_id"
            print(msg)
            diag.missing_sentence_id += 1
            diag.rejection_logs.append(msg)
            skipped_count += 1
            continue
        s_id = str(s_id_raw).strip()

        speaker_id_raw = s.get("speakerId")
        if not speaker_id_raw or not str(speaker_id_raw).strip():
            msg = f"[CTC_DATASET] rejected sample {s_id} reason=missing_speaker_id"
            print(msg)
            diag.missing_speaker_id += 1
            diag.rejection_logs.append(msg)
            skipped_count += 1
            continue
        speaker_id = str(speaker_id_raw).strip()

        session_id_raw = s.get("sessionId")
        if not session_id_raw or not str(session_id_raw).strip():
            msg = f"[CTC_DATASET] rejected sample {s_id} reason=missing_session_id"
            print(msg)
            diag.missing_session_id += 1
            diag.rejection_logs.append(msg)
            skipped_count += 1
            continue
        session_id = str(session_id_raw).strip()

        recording_id_raw = s.get("recordingId")
        if not recording_id_raw or not str(recording_id_raw).strip():
            msg = f"[CTC_DATASET] rejected sample {s_id} reason=missing_recording_id"
            print(msg)
            diag.missing_recording_id += 1
            diag.rejection_logs.append(msg)
            skipped_count += 1
            continue
        recording_id = str(recording_id_raw).strip()
        base_recording_id = strip_augmentation_suffix(recording_id)

        raw_glosses = s.get("glossSequence", [])
        if isinstance(raw_glosses, str):
            raw_glosses = canonicalize_gloss_sequence(raw_glosses)
        elif isinstance(raw_glosses, (list, tuple)):
            raw_glosses = [g for g in raw_glosses if str(g).strip()]
        else:
            raw_glosses = []

        frames = s.get("frames", [])
        if not raw_glosses or not isinstance(frames, list) or len(frames) == 0:
            msg = f"[CTC_DATASET] rejected sample {s_id} reason=empty_gloss_or_frames"
            print(msg)
            diag.empty_gloss_or_frames += 1
            diag.rejection_logs.append(msg)
            skipped_count += 1
            continue

        # A. Vocabulary validation
        try:
            target_ids = vocab.encode(raw_glosses)
        except ValueError as e:
            msg = f"[CTC_DATASET] rejected sample {s_id} reason=unknown_gloss: {e}"
            print(msg)
            diag.unknown_gloss += 1
            diag.rejection_logs.append(msg)
            skipped_count += 1
            continue

        if len(target_ids) == 0:
            msg = f"[CTC_DATASET] rejected sample {s_id} reason=empty_gloss_or_frames"
            print(msg)
            diag.empty_gloss_or_frames += 1
            diag.rejection_logs.append(msg)
            skipped_count += 1
            continue

        # B. Canonical 212-D frame validation & NaN/Inf sanitization
        feature_ver = s.get("featureVersion")
        is_explicit_canonical = feature_ver in COMPATIBLE_FEATURE_VERSIONS

        feats: List[List[float]] = []
        is_canonical = True
        tracker = create_hand_track_state()

        for f in frames:
            if not isinstance(f, dict):
                is_canonical = False
                break
            vec_212 = _extract_frame_212_vector(f, is_explicit_canonical, tracker)
            if vec_212 is None or len(vec_212) != CTC_FEATURE_DIM:
                is_canonical = False
                break
            feats.append(vec_212)

        if not is_canonical or len(feats) < min_sequence_length:
            msg = f"[CTC_DATASET] rejected sample {s_id} reason=missing_or_invalid_canonical_212"
            print(msg)
            diag.invalid_canonical_212 += 1
            diag.rejection_logs.append(msg)
            skipped_count += 1
            continue

        t_len = len(feats)
        target_len = len(target_ids)

        # C. Strict CTC Length Validation (P0-4)
        minimum_required_t, _ = compute_ctc_minimum_required_length(target_ids)
        if t_len < minimum_required_t:
            msg = (
                f"[CTC_DATASET] rejected sample {s_id} "
                f"reason=ctc_input_too_short (T={t_len} < minRequired={minimum_required_t})"
            )
            print(msg)
            diag.ctc_input_too_short += 1
            diag.rejection_logs.append(msg)
            skipped_count += 1
            continue

        x_tensor = torch.tensor(feats, dtype=torch.float32)    # [T, 212]
        y_tensor = torch.tensor(target_ids, dtype=torch.long)  # [U]

        valid_samples.append({
            "id": s_id,
            "sentence_id": s_id,
            "speaker_id": speaker_id,
            "session_id": session_id,
            "recording_id": recording_id,
            "base_recording_id": base_recording_id,
            "split": str(s.get("split", "")).strip().lower(),
            "is_augmented": bool(s.get("isAugmented", False) or (base_recording_id != recording_id)),
            "features": x_tensor,
            "targets": y_tensor,
            "glosses": [vocab.id_to_word[tid] for tid in target_ids],
            "T": t_len,
            "U": target_len,
        })

    diag.valid_samples = len(valid_samples)
    diag.rejected_samples = skipped_count

    print("==================================================")
    print("DATASET VALIDATION REPORT")
    print("==================================================")
    print(f"Total samples: {len(raw_samples)}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Rejected samples: {skipped_count}")
    print(f"Missing speakerId: {diag.missing_speaker_id}")
    print(f"Missing sessionId: {diag.missing_session_id}")
    print(f"Missing recordingId: {diag.missing_recording_id}")
    print(f"Missing sentenceId: {diag.missing_sentence_id}")
    print("==================================================")

    if return_diagnostics:
        return valid_samples, skipped_count, diag
    return valid_samples, skipped_count


# =============================================================================
# 2. LEAKAGE-FREE GROUP SPLITTING (SPEAKER -> SESSION -> BASE RECORDING)
# =============================================================================

def split_dataset_by_group(
    samples: Sequence[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into Train / Validation / Test by independent group hierarchy (P0-2):
    1. `speakerId` (if >= 3 unique speakers) -> Speaker-Independent evaluation.
    2. `sessionId` (if >= 3 unique sessions) -> Session-Independent evaluation.
    3. `base_recording_id` / `recordingId` (if >= 3 unique base recordings) -> Recording-Independent.
    4. If < 3 independent groups exist at all levels, raises `ValueError` (Fail Fast).
    Also strips augmentation suffixes from `recording_id` so augmented copies of a recording
    are always bound to the same group and never leak into Validation or Test sets.
    """
    if len(samples) == 0:
        return [], [], []

    def get_base_rec(s: Dict[str, Any]) -> str:
        if "base_recording_id" in s and s["base_recording_id"]:
            return str(s["base_recording_id"])
        return strip_augmentation_suffix(str(s["recording_id"]))

    all_speakers = set(s["speaker_id"] for s in samples)
    all_sessions = set(s["session_id"] for s in samples)
    all_recordings = set(get_base_rec(s) for s in samples)

    split_level = "speakerId"
    if len(all_speakers) >= 3:
        split_level = "speakerId"
        group_key_func = lambda s: s["speaker_id"]
    else:
        if len(all_speakers) > 0:
            print(
                f"[CTC_SPLIT] Speaker-independent strategy rejected. "
                f"strategy=speaker unique_groups={len(all_speakers)} required_minimum=3"
            )
        if len(all_sessions) >= 3:
            split_level = "sessionId"
            group_key_func = lambda s: s["session_id"]
        else:
            if len(all_sessions) > 0:
                print(
                    f"[CTC_SPLIT] Session strategy rejected. "
                    f"strategy=session unique_groups={len(all_sessions)} required_minimum=3"
                )
            if len(all_recordings) >= 3:
                split_level = "recordingId"
                group_key_func = get_base_rec
            else:
                split_level = "recordingId"
                raise ValueError(
                    f"[CTC_SPLIT] Not enough independent groups. strategy={split_level} "
                    f"unique_groups={len(all_recordings)} required_minimum=3. "
                    f"Need at least 3 independent groups to create non-empty train/validation/test sets."
                )

    group_map: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        g = group_key_func(s)
        group_map.setdefault(g, []).append(s)

    unique_groups = sorted(list(group_map.keys()))

    # Respect explicit train/val/test group designations when present (e.g. from keypoints_splited/{train,test})
    explicit_tr = {
        g for g, items in group_map.items()
        if any(it.get("split") == "train" for it in items) or "train" in g.lower()
    }
    explicit_va = {
        g for g, items in group_map.items()
        if any(it.get("split") == "val" for it in items) or "val" in g.lower()
    }
    explicit_te = {
        g for g, items in group_map.items()
        if any(it.get("split") == "test" for it in items) or "test" in g.lower()
    }

    if explicit_tr and explicit_va and explicit_te and not (
        (explicit_tr & explicit_va) or (explicit_tr & explicit_te) or (explicit_va & explicit_te)
    ):
        train_groups = explicit_tr
        val_groups = explicit_va
        test_groups = explicit_te
    else:
        rng = random.Random(seed)
        rng.shuffle(unique_groups)

        n_groups = len(unique_groups)
        n_train_g = min(n_groups - 2, max(1, int(n_groups * train_ratio)))
        n_val_g = min(n_groups - n_train_g - 1, max(1, int(n_groups * val_ratio)))

        train_groups = set(unique_groups[:n_train_g])
        val_groups = set(unique_groups[n_train_g : n_train_g + n_val_g])
        test_groups = set(unique_groups[n_train_g + n_val_g :])

    # Zero-Leakage Group Overlap Guard
    if (train_groups & val_groups) or (train_groups & test_groups) or (val_groups & test_groups):
        raise ValueError(f"FATAL: Overlap detected in dataset split at level '{split_level}'!")

    train_s: List[Dict[str, Any]] = []
    val_s: List[Dict[str, Any]] = []
    test_s: List[Dict[str, Any]] = []

    for g, items in group_map.items():
        if g in train_groups:
            train_s.extend(items)
        elif g in val_groups:
            # Exclude synthetic/augmented variants from validation if original recordings exist in the group
            orig_items = [it for it in items if not it.get("is_augmented", False)]
            val_s.extend(orig_items if orig_items else items)
        else:
            orig_items = [it for it in items if not it.get("is_augmented", False)]
            test_s.extend(orig_items if orig_items else items)

    # Secondary check: ensure no base_recording_id is shared across splits even when splitting by speaker/session
    train_base_recs = set(get_base_rec(s) for s in train_s)
    val_base_recs = set(get_base_rec(s) for s in val_s)
    test_base_recs = set(get_base_rec(s) for s in test_s)
    if (train_base_recs & val_base_recs) or (train_base_recs & test_base_recs) or (val_base_recs & test_base_recs):
        raise ValueError("FATAL: Base recording leakage detected across train/validation/test splits!")

    is_speaker_independent = split_level == "speakerId"

    print("\n==================================================")
    print("DATASET SPLIT REPORT")
    print("==================================================")
    print(f"Total samples: {len(samples)}")
    print(f"Unique speakers: {len(all_speakers)}")
    print(f"Unique sessions: {len(all_sessions)}")
    print(f"Unique recordings: {len(all_recordings)}")
    print(f"Split Strategy Level: {split_level} (Speaker-Independent: {is_speaker_independent})")
    print("\nTrain:")
    print(f"  samples = {len(train_s)}")
    print(f"  speakers = {len(set(s['speaker_id'] for s in train_s))}")
    print(f"  sessions = {len(set(s['session_id'] for s in train_s))}")
    print("\nValidation:")
    print(f"  samples = {len(val_s)}")
    print(f"  speakers = {len(set(s['speaker_id'] for s in val_s))}")
    print(f"  sessions = {len(set(s['session_id'] for s in val_s))}")
    print("\nTest:")
    print(f"  samples = {len(test_s)}")
    print(f"  speakers = {len(set(s['speaker_id'] for s in test_s))}")
    print(f"  sessions = {len(set(s['session_id'] for s in test_s))}")
    print("==================================================\n")

    return train_s, val_s, test_s


def pad_collate_fn(
    batch: Sequence[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Collate variable-length `[T_i, 212]` feature sequences into a padded batch `[B, T_max, 212]`,
    preserving true `input_lengths` and `target_lengths` for `torch.nn.CTCLoss`.
    """
    features = [item["features"] for item in batch]
    targets = [item["targets"] for item in batch]
    input_lengths = torch.tensor([item["T"] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item["U"] for item in batch], dtype=torch.long)
    glosses = [item["glosses"] for item in batch]

    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    concatenated_targets = torch.cat(targets, dim=0)

    return padded_features, concatenated_targets, input_lengths, target_lengths, glosses


def load_continuous_json_dataset(
    dataset_path: str,
    vocab: Optional[Vocabulary] = None,
    min_sequence_length: int = 3,
) -> Tuple[List[Dict[str, Any]], int, DatasetValidationDiagnostic, Vocabulary]:
    """Load and validate a continuous sentence JSON corpus from disk."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Continuous dataset file not found at '{dataset_path}'")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        raw_samples = raw_data
    elif isinstance(raw_data, dict):
        raw_samples = raw_data.get("samples", [])
    else:
        raise ValueError(f"Unsupported JSON root structure in '{dataset_path}'")

    if vocab is None:
        vocab = Vocabulary()

    valid_samples, skipped_count, diag = parse_and_validate_samples(
        raw_samples,
        vocab,
        min_sequence_length=min_sequence_length,
        return_diagnostics=True,
    )
    return valid_samples, skipped_count, diag, vocab


# =============================================================================
# 3. SINGLE-SIGN BASELINE DATASET LOADER & PRE-AUGMENTATION SPLIT (138-D, 30f)
# =============================================================================

def parse_recording_identity(row: Dict[str, Any], clean_label: str) -> Tuple[str, str, bool]:
    """
    Exact reproduction of `parseRecordingIdentity` in `src/utils/lstmModel.ts`:
    Identifies whether a row belongs to an augmented sequence and recovers its
    canonical `originalRecordingId` (`<label>::<base_seq_id>`).
    """
    raw_seq = str(row.get("sequence_id") or row.get("video_file") or "base").strip()
    aug_type = str(row.get("aug_type") or "original").strip().lower()

    stripped_base = strip_augmentation_suffix(raw_seq)
    has_aug_suffix = stripped_base != raw_seq
    try:
        is_aug_num = int(float(row.get("is_augmented", 0))) == 1
    except (TypeError, ValueError):
        is_aug_num = False

    is_augmented = is_aug_num or has_aug_suffix or (aug_type not in ("", "original", "none", "raw_live"))
    original_recording_id = f"{clean_label}::{stripped_base or 'base'}"
    variant_key = (
        f"{original_recording_id}::aug::{raw_seq}_{aug_type}"
        if is_augmented
        else f"{original_recording_id}::orig::{raw_seq}"
    )
    return original_recording_id, variant_key, is_augmented


def is_valid_dataset_row_landmarks(row: Dict[str, Any]) -> bool:
    """Exact reproduction of `isValidDatasetRowLandmarks` in `src/utils/lstmModel.ts`."""
    if not isinstance(row, dict):
        return False
    non_zero = 0
    for i in range(21):
        try:
            x = float(row.get(f"x_{i}", 0.0))
            y = float(row.get(f"y_{i}", 0.0))
            z = float(row.get(f"z_{i}", 0.0))
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False
        if abs(x) > 1e-6 or abs(y) > 1e-6 or abs(z) > 1e-6:
            non_zero += 1
    if non_zero < 10:
        return False
    dx9 = float(row.get("x_9", 0.0)) - float(row.get("x_0", 0.0))
    dy9 = float(row.get("y_9", 0.0)) - float(row.get("y_0", 0.0))
    dz9 = float(row.get("z_9", 0.0)) - float(row.get("z_0", 0.0))
    palm_dist = math.sqrt(dx9 * dx9 + dy9 * dy9 + dz9 * dz9)
    return palm_dist >= 1e-4


def prepare_single_sign_sequences(
    rows: Sequence[Dict[str, Any]],
    min_frames_per_class: int = 2,
    min_valid_frames_per_recording: int = 2,
    max_invalid_frame_ratio: float = 0.5,
) -> Dict[str, Any]:
    """
    Exact reproduction of `prepareSequences` in `src/utils/lstmModel.ts`:
    Groups frame rows by recording variant, trims idle boundaries, extracts 138-D
    feature vectors, resamples to 30 frames, and applies Pose Modality Dropout / Shortcut
    Learning Protection across classes.
    """
    rejected_recordings: List[Dict[str, Any]] = []
    if not rows:
        return {
            "X": [],
            "Y": [],
            "base_session_ids": [],
            "is_augmented_flags": [],
            "label_map": [],
            "num_classes": 0,
            "rejected_recordings": rejected_recordings,
        }

    counts: Dict[str, int] = {}
    for row in rows:
        lbl = normalize_single_gloss(row.get("label"))
        if lbl and lbl not in ("UNKNOWN", "?"):
            counts[lbl] = counts.get(lbl, 0) + 1

    valid_labels = sorted([lbl for lbl, c in counts.items() if c >= min_frames_per_class])
    label_to_index = {lbl: idx for idx, lbl in enumerate(valid_labels)}

    groups: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        clean_label = normalize_single_gloss(row.get("label"))
        if not clean_label or clean_label in ("UNKNOWN", "?"):
            rejected_recordings.append({
                "recording_id": f"row_{idx}",
                "label": str(row.get("label", "")),
                "reason": "Empty or invalid label (UNKNOWN/?)",
            })
            continue
        if clean_label not in label_to_index:
            continue

        orig_rec_id, variant_key, is_aug = parse_recording_identity(row, clean_label)
        if variant_key not in groups:
            groups[variant_key] = {
                "clean_label": clean_label,
                "original_recording_id": orig_rec_id,
                "is_augmented": is_aug,
                "rows": [],
            }
        groups[variant_key]["rows"].append(row)

    x_data: List[List[List[float]]] = []
    y_data: List[int] = []
    base_session_ids: List[str] = []
    is_augmented_flags: List[bool] = []

    for group_key in sorted(groups.keys()):
        g = groups[group_key]
        g_rows: List[Dict[str, Any]] = g["rows"]
        clean_label = g["clean_label"]
        orig_rec_id = g["original_recording_id"]
        is_aug = g["is_augmented"]

        g_rows.sort(key=lambda r: sanitize_float(r.get("frame_idx", 0), 0.0))
        valid_rows = [r for r in g_rows if is_valid_dataset_row_landmarks(r)]
        invalid_ratio = (len(g_rows) - len(valid_rows)) / len(g_rows) if g_rows else 1.0

        if g_rows and invalid_ratio > max_invalid_frame_ratio:
            rejected_recordings.append({
                "recording_id": orig_rec_id,
                "label": clean_label,
                "reason": f"Invalid frame ratio {invalid_ratio:.2f} > {max_invalid_frame_ratio}",
            })
            continue

        if len(valid_rows) < min_valid_frames_per_recording:
            rejected_recordings.append({
                "recording_id": orig_rec_id,
                "label": clean_label,
                "reason": f"Too few valid frames ({len(valid_rows)} < {min_valid_frames_per_recording})",
            })
            continue

        active_rows = trim_sequence_idle_boundaries(valid_rows)
        raw_vecs: List[List[float]] = []
        for r in active_rows:
            lms = [
                Landmark3D(
                    sanitize_float(r.get(f"x_{i}", 0.0)),
                    sanitize_float(r.get(f"y_{i}", 0.0)),
                    sanitize_float(r.get(f"z_{i}", 0.0)),
                )
                for i in range(21)
            ]
            nmm_vec: Optional[List[float]] = None
            if any(f"nmm_{k}" in r for k in (0, 1, 5)):
                nmm_vec = [sanitize_float(r.get(f"nmm_{k}", 0.0)) for k in range(32)]
            vec_138 = extract_invariant_feature_vector(
                lms,
                velocity=sanitize_float(r.get("velocity", 0.0)),
                delta_x=sanitize_float(r.get("delta_x", 0.0)),
                delta_y=sanitize_float(r.get("delta_y", 0.0)),
                delta_z=sanitize_float(r.get("delta_z", 0.0)),
                pose_or_nmm=nmm_vec,
                include_nmm=True,
            )
            raw_vecs.append(vec_138)

        resampled = resample_sequence(raw_vecs, SINGLE_SIGN_SEQUENCE_LENGTH, SINGLE_SIGN_FEATURE_DIM)
        x_data.append(resampled)
        y_data.append(label_to_index[clean_label])
        base_session_ids.append(orig_rec_id)
        is_augmented_flags.append(is_aug)

    # Shortcut Learning Protection for Pose channels [106..137]
    classes_with_pose = set()
    for seq, lbl_idx in zip(x_data, y_data):
        if any(abs(frame[d]) > 1e-5 for frame in seq for d in range(106, SINGLE_SIGN_FEATURE_DIM)):
            classes_with_pose.add(lbl_idx)

    every_class_has_pose = len(valid_labels) > 0 and len(classes_with_pose) == len(valid_labels)
    for i, seq in enumerate(x_data):
        if (not every_class_has_pose) or (is_augmented_flags[i] and i % 2 == 1):
            for frame in seq:
                for d in range(106, SINGLE_SIGN_FEATURE_DIM):
                    frame[d] = 0.0

    return {
        "X": x_data,
        "Y": y_data,
        "base_session_ids": base_session_ids,
        "is_augmented_flags": is_augmented_flags,
        "label_map": valid_labels,
        "num_classes": len(valid_labels),
        "rejected_recordings": rejected_recordings,
    }


def split_single_sign_dataset(
    rows: Sequence[Dict[str, Any]],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Pre-augmentation recording split for the Single-Sign Baseline (`auditAndSplitDataset` in `lstmModel.ts`).
    Ensures augmented variants stay with their original recording in Train, and Validation/Test
    contain only unaugmented original recordings from independent recording IDs.
    """
    prep = prepare_single_sign_sequences(rows)
    x_all = prep["X"]
    y_all = prep["Y"]
    base_ids = prep["base_session_ids"]
    aug_flags = prep["is_augmented_flags"]
    label_map = prep["label_map"]
    num_classes = prep["num_classes"]

    by_class: Dict[int, Dict[str, Dict[str, List[int]]]] = {}
    for idx, (lbl_idx, rec_id, is_aug) in enumerate(zip(y_all, base_ids, aug_flags)):
        class_map = by_class.setdefault(lbl_idx, {})
        rec_group = class_map.setdefault(rec_id, {"orig": [], "aug": [], "all": []})
        rec_group["all"].append(idx)
        if is_aug:
            rec_group["aug"].append(idx)
        else:
            rec_group["orig"].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    train_recs = set()
    val_recs = set()
    test_recs = set()

    for lbl_idx in range(num_classes):
        class_map = by_class.get(lbl_idx, {})
        sorted_rec_ids = sorted(class_map.keys())
        rng = random.Random(seed + lbl_idx * 997)
        rng.shuffle(sorted_rec_ids)

        eligible_orig = [rid for rid in sorted_rec_ids if len(class_map[rid]["orig"]) > 0]
        aug_only = [rid for rid in sorted_rec_ids if len(class_map[rid]["orig"]) == 0]

        n_orig = len(eligible_orig)
        if n_orig >= 3:
            n_val = max(1, int(n_orig * 0.2))
            n_test = max(1, int(n_orig * 0.2))
            val_rids = eligible_orig[:n_val]
            test_rids = eligible_orig[n_val : n_val + n_test]
            train_rids = eligible_orig[n_val + n_test :] + aug_only
        elif n_orig == 2:
            val_rids = [eligible_orig[0]]
            test_rids = []
            train_rids = [eligible_orig[1]] + aug_only
        else:
            val_rids = []
            test_rids = []
            train_rids = eligible_orig + aug_only

        for rid in train_rids:
            train_recs.add(rid)
            train_idx.extend(class_map[rid]["all"])
        for rid in val_rids:
            val_recs.add(rid)
            val_idx.extend(class_map[rid]["orig"])
        for rid in test_rids:
            test_recs.add(rid)
            test_idx.extend(class_map[rid]["orig"])

    if (train_recs & val_recs) or (train_recs & test_recs) or (val_recs & test_recs):
        raise ValueError("FATAL: Recording leakage detected in single-sign split!")

    return {
        "label_map": label_map,
        "num_classes": num_classes,
        "train_X": [x_all[i] for i in train_idx],
        "train_Y": [y_all[i] for i in train_idx],
        "val_X": [x_all[i] for i in val_idx],
        "val_Y": [y_all[i] for i in val_idx],
        "test_X": [x_all[i] for i in test_idx],
        "test_Y": [y_all[i] for i in test_idx],
        "train_recording_ids": sorted(train_recs),
        "val_recording_ids": sorted(val_recs),
        "test_recording_ids": sorted(test_recs),
        "rejected_recordings": prep["rejected_recordings"],
    }


def load_single_sign_rows(file_path: str) -> List[Dict[str, Any]]:
    """Load single-sign `DatasetRow` records from CSV or JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Single-sign dataset file not found at '{file_path}'")
    if file_path.lower().endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("rows", [])
    rows: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows
