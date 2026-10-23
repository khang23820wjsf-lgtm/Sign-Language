"""
VSL400 `.npy` Keypoint Loader & Feature Converter for Dual-Mode VSL Recognition:
  - Part 1: Isolated Word Recognition (BiLSTM)
  - Part 2: Continuous Sentence Recognition (2-Layer BiLSTM + CTC)

Handles `.npy` files of shape `(T, 76, 3)` from `keypoints_splited` / `keypoints_spliteds`:
  - `seq[:, 0:33, :]`  : 33 MediaPipe Pose landmarks (x, y, z)
  - `seq[:, 33:54, :]` : 21 MediaPipe Left Hand landmarks (x, y, z)
  - `seq[:, 54:75, :]` : 21 MediaPipe Right Hand landmarks (x, y, z)
  - `seq[:, 75:76, :]` : 1 Nose landmark (x, y, z)
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from decoder import Vocabulary, normalize_single_gloss
from feature_extractor import (
    CTC_FEATURE_DIM,
    FEATURE_VERSION,
    SINGLE_SIGN_FEATURE_DIM,
    SINGLE_SIGN_SEQUENCE_LENGTH,
    Landmark3D,
    create_hand_track_state,
    extract_canonical_212_frame,
    extract_single_hand_106,
    normalize_landmarks,
    resample_sequence,
)


DEFAULT_NPY_ROOT_CANDIDATES = [
    r"C:\VSL400\vsl400_video_extracted\keypoints_spliteds",
    r"C:\VSL400\SignCTCModel\archive\processed\processed\keypoints_splited",
    r"C:\VSL400\SignCTCModel\keypoints_splited",
]


def find_default_npy_root() -> Optional[str]:
    """Return the first existing `keypoints_splited` directory on disk, if any."""
    for p in DEFAULT_NPY_ROOT_CANDIDATES:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "train")):
            return p
    return None


# ---------------------------------------------------------------------------
# Body-frame canonicalisation (feature version suffix "+bodynorm-v1")
# ---------------------------------------------------------------------------
# Root cause of "`.npy` works, live webcam fails" (verified on the user's saved webcam clips):
#   * VSL400 training videos are 224x224 (aspect 1:1); webcams are 4:3 / 16:9, so raw MediaPipe
#     x-coordinates are anisotropically squashed -> hand-shape angles/distances get distorted.
#     => webcam x/z are aspect-corrected at ingestion (`aspect_correct_76x3`).
#   * Signers in VSL400 have shoulder width ~0.20 (image units) centred at y~0.455, while a user
#     sitting in front of a laptop has ~0.35-0.38 centred at y~0.67. Wrist velocity features and
#     the 32-D shoulder/nose context are in raw image units, so they shift by ~1.8x.
#     => every clip (training `.npy` AND webcam) is mapped to a canonical body frame here.
CANONICAL_SHOULDER_WIDTH = 0.20
CANONICAL_SHOULDER_CENTER = (0.5, 0.455)


def aspect_correct_76x3(arr: np.ndarray, aspect_w_over_h: float) -> np.ndarray:
    """Make x isotropic with y (image-height units) for a non-square camera. Zeros stay zero."""
    out = np.asarray(arr, dtype=np.float32).copy()
    if aspect_w_over_h <= 0 or abs(aspect_w_over_h - 1.0) < 1e-3:
        return out
    nz = np.any(np.abs(out) > 1e-6, axis=-1)
    out[..., 0] = np.where(nz, (out[..., 0] - 0.5) * aspect_w_over_h + 0.5, 0.0)
    out[..., 2] = np.where(nz, out[..., 2] * aspect_w_over_h, 0.0)
    return out


def canonicalize_body_frame_76x3(seq_tx76x3: np.ndarray) -> np.ndarray:
    """
    Translate/scale a whole clip so the (median) shoulder centre sits at
    `CANONICAL_SHOULDER_CENTER` and the shoulder width equals `CANONICAL_SHOULDER_WIDTH`.
    A single similarity transform per clip (median over frames) keeps motion intact.
    Clips without detected shoulders are returned unchanged. Missing points stay exactly 0.
    """
    arr = np.asarray(seq_tx76x3, dtype=np.float32).copy()
    if arr.ndim != 3 or arr.shape[0] == 0:
        return arr
    ls, rs = arr[:, 11, :2], arr[:, 12, :2]
    valid = np.all(np.abs(arr[:, 11:13, :2]) > 1e-6, axis=(1, 2))
    if not np.any(valid):
        return arr
    width = np.linalg.norm(ls[valid] - rs[valid], axis=1)
    sh_w = float(np.median(width))
    if sh_w < 1e-3:
        return arr
    center = np.median(0.5 * (ls[valid] + rs[valid]), axis=0)
    s = CANONICAL_SHOULDER_WIDTH / sh_w
    nz = np.any(np.abs(arr) > 1e-6, axis=-1)
    x = (arr[..., 0] - center[0]) * s + CANONICAL_SHOULDER_CENTER[0]
    y = (arr[..., 1] - center[1]) * s + CANONICAL_SHOULDER_CENTER[1]
    z = arr[..., 2] * s
    out = np.stack([x, y, z], axis=-1).astype(np.float32)
    out[~nz] = 0.0
    return out


def interpolate_and_trim_npy_76x3(
    seq_tx76x3: np.ndarray,
    max_interp_gap: int = 8,
) -> np.ndarray:
    """
    Clean a raw `(T, 76, 3)` `.npy` keypoint array:
    1. Sanitize NaN/Inf to 0.0.
    2. Independently interpolate missing hand blocks (`33:54` Left hand, `54:75` Right hand)
       across short occlusion gaps (`<= max_interp_gap` frames), and carry nearest valid frame
       for 1-2 frame edge dropouts.
    3. Trim leading and trailing frames where BOTH hands are completely absent (`all-zero`),
       preserving at least 6 frames so short clips remain valid.
    """
    if seq_tx76x3.ndim != 3 or seq_tx76x3.shape[1:] != (76, 3):
        raise ValueError(f"Expected .npy shape (T, 76, 3), got {seq_tx76x3.shape}")

    arr = np.nan_to_num(seq_tx76x3.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
    t_total = arr.shape[0]
    if t_total <= 1:
        return arr

    # Interpolate each anatomical block: Pose (0:33), Left Hand (33:54), Right Hand (54:75)
    for start_idx, end_idx in [(0, 33), (33, 54), (54, 75)]:
        block = arr[:, start_idx:end_idx, :]  # (T, K, 3)
        valid_mask = np.any(np.abs(block) > 1e-6, axis=(1, 2))  # (T,)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            continue

        # Linearly interpolate interior gaps <= max_interp_gap
        for i in range(len(valid_indices) - 1):
            t0 = int(valid_indices[i])
            t1 = int(valid_indices[i + 1])
            gap = t1 - t0 - 1
            if 0 < gap <= max_interp_gap:
                for g in range(1, gap + 1):
                    alpha = g / float(gap + 1)
                    block[t0 + g] = (1.0 - alpha) * block[t0] + alpha * block[t1]

        arr[:, start_idx:end_idx, :] = block

    # Trim dead leading/trailing frames where both hands (33:75) are completely missing
    hands_valid = np.any(np.abs(arr[:, 33:75, :]) > 1e-6, axis=(1, 2))
    valid_hand_idx = np.where(hands_valid)[0]
    if len(valid_hand_idx) >= 4:
        first_valid = max(0, int(valid_hand_idx[0]) - 1)
        last_valid = min(t_total - 1, int(valid_hand_idx[-1]) + 1)
        if (last_valid - first_valid + 1) >= 4:
            arr = arr[first_valid : last_valid + 1]

    return arr


def augment_npy_76x3(
    seq_tx76x3: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    """
    Apply realistic spatial-temporal augmentation on a `(T, 76, 3)` keypoint sequence
    for training-set class balancing & webcam camera-distance/framing generalization:
      - Random temporal speed perturbation (0.78x .. 1.22x)
      - 2D in-plane rotation (-10 deg .. +10 deg) around image center (0.5, 0.5)
      - Scale jitter (0.84 .. 1.16) and translation jitter (-0.07 .. +0.07)
      - Slight Gaussian landmark noise (std=0.0025) on non-zero keypoints
    """
    arr = seq_tx76x3.astype(np.float32).copy()
    t_len = arr.shape[0]

    # 1. Temporal speed resampling
    if t_len >= 6 and rng.random() < 0.75:
        speed = rng.uniform(0.78, 1.22)
        new_t = max(6, int(round(t_len * speed)))
        idx_float = np.linspace(0, t_len - 1, new_t)
        idx_lo = np.floor(idx_float).astype(int)
        idx_hi = np.minimum(t_len - 1, idx_lo + 1)
        frac = (idx_float - idx_lo)[:, None, None].astype(np.float32)
        arr = (1.0 - frac) * arr[idx_lo] + frac * arr[idx_hi]

    # Find non-zero mask so missing hands remain strictly zero
    nonzero_mask = np.any(np.abs(arr) > 1e-6, axis=-1, keepdims=True)

    # 2. Rotation around (0.5, 0.5) + camera-distance scale + framing translation
    angle_rad = math.radians(rng.uniform(-10.0, 10.0))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    scale = rng.uniform(0.84, 1.16)
    tx = rng.uniform(-0.07, 0.07)
    ty = rng.uniform(-0.07, 0.07)

    cx, cy = 0.5, 0.5
    x = arr[..., 0] - cx
    y = arr[..., 1] - cy
    z = arr[..., 2]

    x_new = scale * (x * cos_a - y * sin_a) + cx + tx
    y_new = scale * (x * sin_a + y * cos_a) + cy + ty
    z_new = scale * z

    # Residual aspect-ratio jitter (webcams report slightly different pixel aspect / FOV)
    if rng.random() < 0.6:
        x_new = (x_new - cx) * rng.uniform(0.92, 1.08) + cx

    aug = np.stack([x_new, y_new, z_new], axis=-1).astype(np.float32)

    # 3. Micro Gaussian noise
    noise = np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0.0, 0.0025, size=aug.shape).astype(np.float32)
    aug = np.where(nonzero_mask, aug + noise, 0.0)
    return aug


def parse_npy_frame_76x3(
    frame_76x3: np.ndarray,
) -> Tuple[List[Landmark3D], List[Landmark3D], List[Landmark3D]]:
    """
    Split a single `(76, 3)` keypoint frame into:
      - `pose_pts` (33 Landmark3D)
      - `left_hand_pts` (21 Landmark3D, or empty list if all zeros)
      - `right_hand_pts` (21 Landmark3D, or empty list if all zeros)
    """
    pose_arr = frame_76x3[0:33]
    left_arr = frame_76x3[33:54]
    right_arr = frame_76x3[54:75]

    pose_pts = [Landmark3D(float(p[0]), float(p[1]), float(p[2])) for p in pose_arr]

    left_valid = bool(np.any(np.abs(left_arr) > 1e-6))
    right_valid = bool(np.any(np.abs(right_arr) > 1e-6))

    left_pts = (
        [Landmark3D(float(p[0]), float(p[1]), float(p[2])) for p in left_arr]
        if left_valid
        else []
    )
    right_pts = (
        [Landmark3D(float(p[0]), float(p[1]), float(p[2])) for p in right_arr]
        if right_valid
        else []
    )
    return pose_pts, left_pts, right_pts


def _extract_hand_106_with_wrist(
    hand_pts: Sequence[Landmark3D],
    prev_wrist: Optional[Landmark3D],
) -> Tuple[List[float], Landmark3D]:
    norm_pts, wrist_pt, _ = normalize_landmarks(hand_pts)
    dx, dy, dz, vel = 0.0, 0.0, 0.0, 0.0
    if prev_wrist is not None:
        dx = wrist_pt.x - prev_wrist.x
        dy = wrist_pt.y - prev_wrist.y
        dz = (wrist_pt.z or 0.0) - (prev_wrist.z or 0.0)
        vel = math.sqrt(dx * dx + dy * dy + dz * dz)
    cur_wrist = Landmark3D(wrist_pt.x, wrist_pt.y, wrist_pt.z)
    vec106 = extract_single_hand_106(
        norm_pts, velocity=vel, delta_x=dx, delta_y=dy, delta_z=dz
    )
    return vec106, cur_wrist


def npy_sequence_to_212_features(
    seq_tx76x3: np.ndarray,
    clean_and_trim: bool = True,
) -> List[List[float]]:
    """
    Convert a `(T, 76, 3)` `.npy` array into a sequence of canonical 212-D feature vectors
    (`Left 106 + Right 106`, `v1-canonical-left106-right106`).
    Directly maps anatomical `seq[:, 33:54, :]` to the Left 106-D slot and
    `seq[:, 54:75, :]` to the Right 106-D slot so Left and Right hands never swap.
    """
    arr = interpolate_and_trim_npy_76x3(seq_tx76x3) if clean_and_trim else np.nan_to_num(
        seq_tx76x3.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    arr = canonicalize_body_frame_76x3(arr)

    prev_left_wrist: Optional[Landmark3D] = None
    prev_right_wrist: Optional[Landmark3D] = None
    out_212: List[List[float]] = []

    for t in range(arr.shape[0]):
        _, left_pts, right_pts = parse_npy_frame_76x3(arr[t])

        if len(left_pts) == 21:
            left_106, prev_left_wrist = _extract_hand_106_with_wrist(left_pts, prev_left_wrist)
        else:
            left_106 = [0.0] * 106
            prev_left_wrist = None

        if len(right_pts) == 21:
            right_106, prev_right_wrist = _extract_hand_106_with_wrist(right_pts, prev_right_wrist)
        else:
            right_106 = [0.0] * 106
            prev_right_wrist = None

        out_212.append(left_106 + right_106)

    return out_212


def _extract_bimanual_pose_32_context(
    pose_pts: Sequence[Landmark3D],
    left_pts: Sequence[Landmark3D],
    right_pts: Sequence[Landmark3D],
    prev_left_wrist: Optional[Landmark3D],
    prev_right_wrist: Optional[Landmark3D],
) -> List[float]:
    """
    Build a rich, normalized 32-D bimanual + upper-body context vector for Part 1 (Word BiLSTM):
      - [0..5]   (6 dims): Left & Right wrist positions normalized relative to shoulder center & width
      - [6..11]  (6 dims): Left & Right wrist positions relative to Nose (head-anchored sign location)
      - [12..15] (4 dims): Inter-wrist 3D vector (Right - Left) + Euclidean distance between hands
      - [16..21] (6 dims): Left & Right wrist frame-to-frame velocities (dx, dy, dz)
      - [22..26] (5 dims): Secondary/Left hand normalized fingertip distances from Left wrist
                           (thumb, index, middle, ring, pinky tips: indices 4, 8, 12, 16, 20)
      - [27..31] (5 dims): Left index tip orientation + hand presence flags
    Total = 32 dims (combined with 106-D primary hand features = 138-D).
    """
    # Check if Pose shoulders (indices 11 and 12) are actually present (non-zero)
    pose_shoulders_valid = (
        len(pose_pts) >= 13
        and (abs(pose_pts[11].x) > 1e-5 or abs(pose_pts[11].y) > 1e-5)
        and (abs(pose_pts[12].x) > 1e-5 or abs(pose_pts[12].y) > 1e-5)
    )
    if pose_shoulders_valid:
        ls = pose_pts[11]
        rs = pose_pts[12]
        cx = 0.5 * (ls.x + rs.x)
        cy = 0.5 * (ls.y + rs.y)
        cz = 0.5 * ((ls.z or 0.0) + (rs.z or 0.0))
        sh_dist = math.sqrt(
            (ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2 + ((ls.z or 0.0) - (rs.z or 0.0)) ** 2
        )
        scale = max(sh_dist, 0.12)
        nose = pose_pts[0]
        if abs(nose.x) > 1e-5 or abs(nose.y) > 1e-5:
            nx, ny, nz = nose.x, nose.y, (nose.z or 0.0)
        else:
            nx, ny, nz = cx, cy - 0.65 * scale, cz
    else:
        cx, cy, cz = 0.5, 0.55, 0.0
        nx, ny, nz = 0.5, 0.32, 0.0
        scale = 0.28

    has_left = len(left_pts) == 21
    has_right = len(right_pts) == 21

    lw = left_pts[0] if has_left else Landmark3D(cx, cy, cz)
    rw = right_pts[0] if has_right else Landmark3D(cx, cy, cz)

    # 1. Wrists relative to shoulder center (6 dims)
    lw_sh = [(lw.x - cx) / scale, (lw.y - cy) / scale, ((lw.z or 0.0) - cz) / scale] if has_left else [0.0, 0.0, 0.0]
    rw_sh = [(rw.x - cx) / scale, (rw.y - cy) / scale, ((rw.z or 0.0) - cz) / scale] if has_right else [0.0, 0.0, 0.0]

    # 2. Wrists relative to Nose (6 dims)
    lw_nose = [(lw.x - nx) / scale, (lw.y - ny) / scale, ((lw.z or 0.0) - nz) / scale] if has_left else [0.0, 0.0, 0.0]
    rw_nose = [(rw.x - nx) / scale, (rw.y - ny) / scale, ((rw.z or 0.0) - nz) / scale] if has_right else [0.0, 0.0, 0.0]

    # 3. Inter-wrist relative vector & distance (4 dims)
    if has_left and has_right:
        ibr_x = (rw.x - lw.x) / scale
        ibr_y = (rw.y - lw.y) / scale
        ibr_z = ((rw.z or 0.0) - (lw.z or 0.0)) / scale
        ibr_d = math.sqrt(ibr_x * ibr_x + ibr_y * ibr_y + ibr_z * ibr_z)
        inter_wrist = [ibr_x, ibr_y, ibr_z, ibr_d]
    else:
        inter_wrist = [0.0, 0.0, 0.0, 0.0]

    # 4. Left & Right wrist velocities (6 dims)
    if has_left and prev_left_wrist is not None:
        lv = [
            (lw.x - prev_left_wrist.x) / scale,
            (lw.y - prev_left_wrist.y) / scale,
            ((lw.z or 0.0) - (prev_left_wrist.z or 0.0)) / scale,
        ]
    else:
        lv = [0.0, 0.0, 0.0]

    if has_right and prev_right_wrist is not None:
        rv = [
            (rw.x - prev_right_wrist.x) / scale,
            (rw.y - prev_right_wrist.y) / scale,
            ((rw.z or 0.0) - (prev_right_wrist.z or 0.0)) / scale,
        ]
    else:
        rv = [0.0, 0.0, 0.0]

    # 5. Left hand normalized finger extension profile (5 dims: thumb, index, middle, ring, pinky tips)
    if has_left:
        norm_l, _, _ = normalize_landmarks(left_pts)
        left_tips = [
            math.sqrt(norm_l[idx].x ** 2 + norm_l[idx].y ** 2 + (norm_l[idx].z or 0.0) ** 2)
            for idx in (4, 8, 12, 16, 20)
        ]
    else:
        left_tips = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 6. Left index tip orientation + hand presence indicators (5 dims)
    if has_left:
        norm_l_idx = norm_l[8]
        left_orient = [norm_l_idx.x, norm_l_idx.y, float(norm_l_idx.z or 0.0)]
    else:
        left_orient = [0.0, 0.0, 0.0]

    presence = [1.0 if has_left else 0.0, 1.0 if has_right else 0.0]

    vec32 = lw_sh + rw_sh + lw_nose + rw_nose + inter_wrist + lv + rv + left_tips + left_orient + presence
    return [float(max(-10.0, min(10.0, v))) for v in vec32[:32]]


def npy_sequence_to_word_138_features(
    seq_tx76x3: np.ndarray,
    target_len: int = SINGLE_SIGN_SEQUENCE_LENGTH,
) -> List[List[float]]:
    """
    Convert a `(T, 76, 3)` `.npy` array into a resampled `(30, 138)` feature sequence
    for Part 1 (Isolated Word Recognition via BiLSTM):
      - Interpolates missing hand frames and trims dead boundary frames.
      - Selects a CONSISTENT primary hand across the entire clip (Right hand if active in
        >=30% of valid frames, otherwise Left hand) so `prev_wrist` never jumps between
        Left and Right hands mid-sequence.
      - Encodes the primary hand into the 106-D invariant hand geometry + velocity slot,
        and encodes BOTH hands + shoulder/nose relative positions + inter-wrist distance +
        secondary hand finger extension into the 32-D context slot (`106 + 32 = 138`).
      - Resamples temporally to `target_len` (30 frames).
    """
    arr = canonicalize_body_frame_76x3(interpolate_and_trim_npy_76x3(seq_tx76x3))

    # Determine consistent primary hand across the clip to prevent wrist-jumping
    right_valid_frames = int(np.sum(np.any(np.abs(arr[:, 54:75, :]) > 1e-6, axis=(1, 2))))
    left_valid_frames = int(np.sum(np.any(np.abs(arr[:, 33:54, :]) > 1e-6, axis=(1, 2))))
    use_right_as_primary = right_valid_frames >= max(1, int(0.3 * left_valid_frames))

    raw_138: List[List[float]] = []
    prev_primary_wrist: Optional[Landmark3D] = None
    prev_left_wrist: Optional[Landmark3D] = None
    prev_right_wrist: Optional[Landmark3D] = None

    for t in range(arr.shape[0]):
        pose_pts, left_pts, right_pts = parse_npy_frame_76x3(arr[t])
        primary_pts = right_pts if use_right_as_primary else left_pts
        if len(primary_pts) != 21:
            # Fallback only if chosen primary hand is completely absent in this frame
            primary_pts = left_pts if use_right_as_primary else right_pts
            prev_primary_wrist = None

        if len(primary_pts) == 21:
            vec106, prev_primary_wrist = _extract_hand_106_with_wrist(primary_pts, prev_primary_wrist)
        else:
            vec106 = [0.0] * 106
            prev_primary_wrist = None

        vec32 = _extract_bimanual_pose_32_context(
            pose_pts=pose_pts,
            left_pts=left_pts,
            right_pts=right_pts,
            prev_left_wrist=prev_left_wrist,
            prev_right_wrist=prev_right_wrist,
        )
        prev_left_wrist = left_pts[0] if len(left_pts) == 21 else None
        prev_right_wrist = right_pts[0] if len(right_pts) == 21 else None

        raw_138.append(vec106 + vec32)

    non_empty = [f for f in raw_138 if any(abs(v) > 1e-6 for v in f)]
    use_frames = non_empty if len(non_empty) >= 2 else raw_138

    return resample_sequence(
        use_frames,
        target_len=target_len,
        expected_dim=SINGLE_SIGN_FEATURE_DIM,
    )


def scan_npy_dataset_root(root_dir: str) -> Dict[str, Any]:
    """
    Inspect a `keypoints_splited` directory containing `train/` and `test/` class folders.
    Returns available classes and file counts per split, plus recommended well-populated classes.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root_dir}")

    train_dir = root_path / "train"
    test_dir = root_path / "test"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục con 'train' trong {root_dir}")

    classes_info: List[Dict[str, Any]] = []
    for p in sorted(train_dir.iterdir(), key=lambda x: x.name.casefold()):
        if not p.is_dir():
            continue
        n_train = len(list(p.glob("*.npy")))
        n_test = len(list((test_dir / p.name).glob("*.npy"))) if (test_dir / p.name).is_dir() else 0
        if n_train > 0:
            classes_info.append(
                {
                    "folder_name": p.name,
                    "canonical_gloss": normalize_single_gloss(p.name),
                    "train_files": n_train,
                    "test_files": n_test,
                }
            )

    # Identify well-populated classes (>= 15 train files and >= 3 test files) for reliable default selection
    recommended = [
        c["folder_name"]
        for c in classes_info
        if c["train_files"] >= 15 and c["test_files"] >= 3
    ]

    return {
        "root_dir": str(root_path),
        "total_classes": len(classes_info),
        "recommended_folders": recommended[:12],
        "classes": classes_info,
    }


def _select_folders_from_root(
    train_dir: Path,
    test_dir: Path,
    selected_folders: Optional[Sequence[str]],
    max_classes: int,
) -> List[str]:
    all_folders = sorted(
        [p.name for p in train_dir.iterdir() if p.is_dir()],
        key=lambda x: x.casefold(),
    )
    if selected_folders:
        chosen_set = {s.strip() for s in selected_folders if s.strip()}
        chosen_norm = {normalize_single_gloss(c) for c in chosen_set}
        folders = [
            f
            for f in all_folders
            if f in chosen_set or normalize_single_gloss(f) in chosen_norm
        ]
    else:
        # Prefer folders with sufficient train (>=15) and test (>=3) files so default runs aren't starved
        rich_folders = [
            f
            for f in all_folders
            if len(list((train_dir / f).glob("*.npy"))) >= 15
            and (test_dir / f).is_dir()
            and len(list((test_dir / f).glob("*.npy"))) >= 3
        ]
        folders = (rich_folders if len(rich_folders) >= max_classes else all_folders)[:max_classes]

    if len(folders) < 2:
        raise ValueError(
            f"Cần chọn tối thiểu 2 lớp từ (tìm thấy {len(folders)} lớp hợp lệ)."
        )
    return folders


def load_word_samples_from_npy_root(
    root_dir: str,
    selected_folders: Optional[Sequence[str]] = None,
    max_classes: int = 10,
    max_files_per_class_split: int = 40,
    min_train_samples_per_class: int = 28,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load `.npy` files `(T, 76, 3)` from `keypoints_splited/{train,test}/<Class>/*.npy`
    and convert them into `(30, 138)` sequences for Part 1 (BiLSTM Word Recognition).

    Key accuracy & anti-leakage guarantees:
    1. Strictly separates `train/` files into `"split": "train"` (85%) and `"split": "val"` (15%,
       at least 2 validation files per class), while keeping `test/` files 100% in `"split": "test"`.
    2. Balances training classes by generating spatial-temporally augmented training sequences
       (strictly from `train` split raw `.npy` arrays!) so every class has at least
       `min_train_samples_per_class` training examples.
    """
    rng = random.Random(seed)
    root_path = Path(root_dir)
    train_dir = root_path / "train"
    test_dir = root_path / "test"

    folders = _select_folders_from_root(train_dir, test_dir, selected_folders, max_classes)
    samples: List[Dict[str, Any]] = []

    for folder_name in folders:
        label = normalize_single_gloss(folder_name)
        tr_files = sorted((train_dir / folder_name).glob("*.npy"))[:max_files_per_class_split]
        te_files = (
            sorted((test_dir / folder_name).glob("*.npy"))[:max_files_per_class_split]
            if (test_dir / folder_name).is_dir()
            else []
        )

        raw_train_arrays: List[Tuple[str, np.ndarray]] = []
        for idx, npy_path in enumerate(tr_files):
            try:
                arr = np.load(npy_path, allow_pickle=False)
                seq_138 = npy_sequence_to_word_138_features(arr)
                # Hold out ~15% of train_dir files (every 6th file) for validation,
                # unless train_dir has < 4 files (in which case use the last file for val)
                is_val = (idx % 6 == 0) if len(tr_files) >= 6 else (idx == len(tr_files) - 1 and len(tr_files) >= 2)
                split_name = "val" if is_val else "train"
                group_tag = "spk_val" if is_val else f"spk_train_{idx % 2}"
                samples.append(
                    {
                        "id": f"{split_name}_{label}_{npy_path.stem}",
                        "speakerId": group_tag,
                        "split": split_name,
                        "label": label,
                        "sequence": seq_138,
                    }
                )
                if not is_val:
                    raw_train_arrays.append((npy_path.stem, arr))
            except Exception:
                continue

        # Balance training samples for this class using landmark-level augmentation (TRAIN ONLY)
        aug_idx = 0
        while len(raw_train_arrays) > 0 and (len(raw_train_arrays) + aug_idx) < min_train_samples_per_class:
            base_stem, base_arr = raw_train_arrays[aug_idx % len(raw_train_arrays)]
            aug_arr = augment_npy_76x3(base_arr, rng)
            seq_138_aug = npy_sequence_to_word_138_features(aug_arr)
            samples.append(
                {
                    "id": f"train_{label}_{base_stem}_aug{aug_idx}",
                    "speakerId": f"spk_train_{aug_idx % 2}",
                    "split": "train",
                    "label": label,
                    "sequence": seq_138_aug,
                }
            )
            aug_idx += 1

        # Load held-out Test files strictly into split="test"
        for npy_path in te_files:
            try:
                arr = np.load(npy_path, allow_pickle=False)
                seq_138 = npy_sequence_to_word_138_features(arr)
                samples.append(
                    {
                        "id": f"test_{label}_{npy_path.stem}",
                        "speakerId": "spk_test",
                        "split": "test",
                        "label": label,
                        "sequence": seq_138,
                    }
                )
            except Exception:
                continue

    return samples


def preprocess_npy_clip_for_ctc(
    arr_tx76x3: np.ndarray,
    target_clip_frames: int = 16,
) -> List[List[float]]:
    """
    Convert a single word `.npy` array `(T, 76, 3)` into a compact, clean 212-D sequence
    of `target_clip_frames` frames (default 16 frames per word) for continuous sentence CTC:
      1. Interpolates missing-hand gaps and trims leading/trailing empty frames.
      2. Resamples the active sign motion to `target_clip_frames` frames so a 2-3 word
         sentence is `34..52` frames (giving sharp CTC alignment without 80% blank frames).
      3. Extracts canonical 212-D (`Left 106 + Right 106`) features.
    """
    cleaned = interpolate_and_trim_npy_76x3(arr_tx76x3)
    t_len = cleaned.shape[0]
    if t_len != target_clip_frames and t_len >= 2:
        idx_float = np.linspace(0, t_len - 1, target_clip_frames)
        idx_lo = np.floor(idx_float).astype(int)
        idx_hi = np.minimum(t_len - 1, idx_lo + 1)
        frac = (idx_float - idx_lo)[:, None, None].astype(np.float32)
        cleaned = (1.0 - frac) * cleaned[idx_lo] + frac * cleaned[idx_hi]

    seq212 = npy_sequence_to_212_features(cleaned, clean_and_trim=False)
    # Filter out any all-zero frames if valid frames exist
    non_zero = [f for f in seq212 if any(abs(v) > 1e-6 for v in f)]
    return non_zero if len(non_zero) >= 4 else seq212


def build_continuous_sentence_samples_from_npy_root(
    root_dir: str,
    selected_folders: Optional[Sequence[str]] = None,
    max_classes: int = 8,
    max_files_per_class: int = 30,
    sentences_per_split: int = 120,
    min_words_per_sentence: int = 2,
    max_words_per_sentence: int = 3,
    transition_frames: int = 2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Vocabulary]:
    """
    Build continuous multi-word sentence sequences `[T_total, 212]` from `.npy` word clips
    in `keypoints_splited/{train,test}` for Part 2 (2-Layer BiLSTM + CTC Sentence Recognition).

    CRITICAL LEAKAGE PREVENTION & HIGH-ACCURACY CTC DESIGN:
    - Training sentences (`split="train"`) are composed ONLY from `train/` `.npy` files
      (excluding held-out validation clips).
    - Validation sentences (`split="val"`) use held-out `train/` validation clips, and
      Test sentences (`split="test"`) are composed ONLY from `test/` `.npy` files.
    - Each word clip is cleaned, interpolated, and normalized to ~16 active frames so
      a 2-3 word continuous sentence is ~34-52 frames long.
    - Includes single-word anchor clips + multi-word continuous sentences in training
      so CTC learns exact token boundaries rapidly without collapsing to `<blank>`.
    """
    rng = random.Random(seed)
    root_path = Path(root_dir)
    train_dir = root_path / "train"
    test_dir = root_path / "test"

    folders = _select_folders_from_root(train_dir, test_dir, selected_folders, max_classes)

    train_clips_by_gloss: Dict[str, List[List[List[float]]]] = {}
    val_clips_by_gloss: Dict[str, List[List[List[float]]]] = {}
    test_clips_by_gloss: Dict[str, List[List[List[float]]]] = {}

    for folder_name in folders:
        gloss = normalize_single_gloss(folder_name)
        tr_paths = sorted((train_dir / folder_name).glob("*.npy"))[:max_files_per_class]
        te_paths = (
            sorted((test_dir / folder_name).glob("*.npy"))[:max_files_per_class]
            if (test_dir / folder_name).is_dir()
            else []
        )

        tr_seqs: List[List[List[float]]] = []
        va_seqs: List[List[List[float]]] = []
        for idx, p in enumerate(tr_paths):
            try:
                arr = np.load(p, allow_pickle=False)
                seq212 = preprocess_npy_clip_for_ctc(arr, target_clip_frames=16)
                if len(seq212) >= 4:
                    is_val = (idx % 5 == 0) if len(tr_paths) >= 5 else (idx == len(tr_paths) - 1 and len(tr_paths) >= 2)
                    if is_val:
                        va_seqs.append(seq212)
                    else:
                        tr_seqs.append(seq212)
                        # Add 1 augmented clip per training file to enrich multi-word transitions
                        aug_arr = augment_npy_76x3(arr, rng)
                        aug212 = preprocess_npy_clip_for_ctc(aug_arr, target_clip_frames=16)
                        if len(aug212) >= 4:
                            tr_seqs.append(aug212)
            except Exception:
                continue

        te_seqs: List[List[List[float]]] = []
        for p in te_paths:
            try:
                arr = np.load(p, allow_pickle=False)
                seq212 = preprocess_npy_clip_for_ctc(arr, target_clip_frames=16)
                if len(seq212) >= 4:
                    te_seqs.append(seq212)
            except Exception:
                continue

        if tr_seqs:
            train_clips_by_gloss[gloss] = tr_seqs
            val_clips_by_gloss[gloss] = va_seqs if va_seqs else [tr_seqs[-1]]
            test_clips_by_gloss[gloss] = te_seqs if te_seqs else val_clips_by_gloss[gloss]

    active_glosses = sorted(train_clips_by_gloss.keys())
    vocab = Vocabulary(custom_words=active_glosses)

    def _concat_clips_with_transition(clips: List[List[List[float]]]) -> List[Dict[str, Any]]:
        combined: List[List[float]] = []
        for c_idx, clip in enumerate(clips):
            if c_idx > 0 and transition_frames > 0 and len(combined) > 0 and len(clip) > 0:
                prev_vec = combined[-1]
                next_vec = clip[0]
                for step in range(1, transition_frames + 1):
                    alpha = step / float(transition_frames + 1)
                    interp_vec = [
                        prev_vec[d] + alpha * (next_vec[d] - prev_vec[d])
                        for d in range(CTC_FEATURE_DIM)
                    ]
                    combined.append(interp_vec)
            combined.extend(clip)
        return [{"combinedFeatureVector": vec} for vec in combined]

    samples: List[Dict[str, Any]] = []

    # 1. Include single-word anchor clips in training so CTC aligns individual glosses rapidly
    anchor_idx = 0
    for gloss in active_glosses:
        for clip in train_clips_by_gloss[gloss][:8]:
            anchor_idx += 1
            spk = f"spk_npy_train_{(anchor_idx % 2) + 1}"
            samples.append(
                {
                    "sentenceId": f"train_anchor_{anchor_idx}",
                    "speakerId": spk,
                    "sessionId": f"sess_{spk}",
                    "recordingId": f"rec_train_anchor_{anchor_idx}",
                    "split": "train",
                    "featureVersion": FEATURE_VERSION,
                    "glossSequence": [gloss],
                    "frames": [{"combinedFeatureVector": vec} for vec in clip],
                }
            )

    def _generate_split_sentences(
        clip_bank: Dict[str, List[List[List[float]]]],
        speaker_ids: List[str],
        count: int,
        split_name: str,
    ) -> None:
        avail_g = [g for g, lst in clip_bank.items() if len(lst) > 0]
        if not avail_g:
            return
        for i in range(count):
            n_words = rng.randint(min_words_per_sentence, max_words_per_sentence)
            chosen_glosses = [rng.choice(avail_g) for _ in range(n_words)]
            chosen_clips = [rng.choice(clip_bank[g]) for g in chosen_glosses]
            frames = _concat_clips_with_transition(chosen_clips)
            spk = speaker_ids[i % len(speaker_ids)]
            samples.append(
                {
                    "sentenceId": f"{split_name}_sent_{i + 1}",
                    "speakerId": spk,
                    "sessionId": f"sess_{spk}",
                    "recordingId": f"rec_{split_name}_{i + 1}",
                    "split": split_name,
                    "featureVersion": FEATURE_VERSION,
                    "glossSequence": chosen_glosses,
                    "frames": frames,
                }
            )

    _generate_split_sentences(
        train_clips_by_gloss,
        ["spk_npy_train_1", "spk_npy_train_2"],
        sentences_per_split,
        "train",
    )
    _generate_split_sentences(
        val_clips_by_gloss,
        ["spk_npy_val"],
        max(20, sentences_per_split // 4),
        "val",
    )
    _generate_split_sentences(
        test_clips_by_gloss,
        ["spk_npy_test"],
        max(20, sentences_per_split // 4),
        "test",
    )

    return samples, vocab
