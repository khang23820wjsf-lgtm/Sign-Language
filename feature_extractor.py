#!/usr/bin/env python3
"""
===============================================================================
FEATURE EXTRACTION, PREPROCESSING & BOUNDARY DETECTION PIPELINE
Exact reproduction of the Vietnamese Sign Language (VSL) feature extraction,
normalization, canonical 212-D hand ordering, occlusion interpolation,
augmentation, and continuous sign boundary detection algorithms.
===============================================================================
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

CTC_FEATURE_DIM: int = 212
SINGLE_HAND_FEATURE_DIM: int = 106
SINGLE_SIGN_FEATURE_DIM: int = 138
NMM_FEATURE_DIM: int = 32
SINGLE_SIGN_SEQUENCE_LENGTH: int = 30
FEATURE_VERSION: str = "v1-canonical-left106-right106"
COMPATIBLE_FEATURE_VERSIONS = frozenset({
    "v1-canonical-left106-right106",
    "ctc-v1-left106-right106",
})
MAX_OCCLUSION_INTERPOLATION_GAP_MS: float = 500.0


# =============================================================================
# 1. COORDINATE UTILITIES & LANDMARK NORMALIZATION
# =============================================================================

@dataclass
class Landmark3D:
    x: float
    y: float
    z: float = 0.0
    visibility: Optional[float] = None

    @classmethod
    def from_any(cls, item: Union["Landmark3D", Dict[str, Any], Sequence[float]]) -> "Landmark3D":
        if isinstance(item, Landmark3D):
            return cls(float(item.x), float(item.y), float(item.z), item.visibility)
        if isinstance(item, dict):
            x = float(item.get("x", 0.0) if item.get("x") is not None else 0.0)
            y = float(item.get("y", 0.0) if item.get("y") is not None else 0.0)
            z = float(item.get("z", 0.0) if item.get("z") is not None else 0.0)
            vis = item.get("visibility")
            return cls(x, y, z, float(vis) if vis is not None else None)
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x = float(item[0])
            y = float(item[1])
            z = float(item[2]) if len(item) >= 3 else 0.0
            return cls(x, y, z)
        raise ValueError(f"Unsupported landmark representation: {type(item)}")

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


def sanitize_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def sanitize_vector(vec: Sequence[Any], expected_dim: Optional[int] = None) -> List[float]:
    cleaned = [sanitize_float(v, 0.0) for v in vec]
    if expected_dim is not None:
        if len(cleaned) < expected_dim:
            cleaned.extend([0.0] * (expected_dim - len(cleaned)))
        elif len(cleaned) > expected_dim:
            cleaned = cleaned[:expected_dim]
    return cleaned


def distance_3d(p1: Landmark3D, p2: Landmark3D) -> float:
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = (p1.z or 0.0) - (p2.z or 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_angle_3d(a: Landmark3D, b: Landmark3D, c: Landmark3D) -> float:
    """Angle at vertex b between vectors ba and bc in radians [0, pi]."""
    v1x = a.x - b.x
    v1y = a.y - b.y
    v1z = (a.z or 0.0) - (b.z or 0.0)

    v2x = c.x - b.x
    v2y = c.y - b.y
    v2z = (c.z or 0.0) - (b.z or 0.0)

    dot = v1x * v2x + v1y * v2y + v1z * v2z
    mag1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    mag2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)

    if mag1 * mag2 < 1e-6:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.acos(cos_val)


def normalize_landmarks(
    raw_landmarks: Sequence[Union[Landmark3D, Dict[str, Any], Sequence[float]]]
) -> Tuple[List[Landmark3D], Landmark3D, float]:
    """
    Exact reproduction of `normalizeLandmarks` in `src/utils/augmentation.ts`:
    1. Translate origin to Wrist (Landmark 0): x' = x - x0, y' = y - y0, z' = z - z0.
    2. Divide by Euclidean distance from Wrist (0) to Middle Finger MCP (9).
       If scaleDistance < 1e-4, fallback to max landmark distance from wrist or 1.0.
    """
    if not raw_landmarks or len(raw_landmarks) != 21:
        raise ValueError(f"Landmarks must have exactly 21 points, got {len(raw_landmarks) if raw_landmarks else 0}")

    pts = [Landmark3D.from_any(pt) for pt in raw_landmarks]
    wrist = Landmark3D(pts[0].x, pts[0].y, pts[0].z)

    translated = [
        Landmark3D(pt.x - wrist.x, pt.y - wrist.y, pt.z - wrist.z)
        for pt in pts
    ]

    middle_mcp = translated[9]
    scale_distance = math.sqrt(
        middle_mcp.x * middle_mcp.x +
        middle_mcp.y * middle_mcp.y +
        middle_mcp.z * middle_mcp.z
    )

    if not math.isfinite(scale_distance) or scale_distance < 1e-4:
        max_dist = max(
            math.sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z)
            for pt in translated
        )
        scale_distance = max_dist if (math.isfinite(max_dist) and max_dist > 0) else 1.0

    normalized = [
        Landmark3D(
            pt.x / scale_distance,
            pt.y / scale_distance,
            pt.z / scale_distance,
        )
        for pt in translated
    ]

    return normalized, wrist, scale_distance


# =============================================================================
# 2. 106-D AND 138-D INVARIANT FEATURE VECTOR EXTRACTION
# =============================================================================

def extract_nmm_feature_vector(
    pose_landmarks: Optional[Sequence[Union[Landmark3D, Dict[str, Any], Sequence[float]]]] = None
) -> List[float]:
    """
    Exact reproduction of `extractNMMFeatureVector` in `src/utils/nmmFeatureExtractor.ts`.
    Extracts 32 body/pose/NMM features normalized by shoulder width and clipped to [-0.15, 0.15].
    """
    if not pose_landmarks or len(pose_landmarks) < 17:
        return [0.0] * NMM_FEATURE_DIM

    pts = [Landmark3D.from_any(p) for p in pose_landmarks]

    def is_reliable(pt: Landmark3D, min_vis: float = 0.55) -> bool:
        if not math.isfinite(pt.x) or not math.isfinite(pt.y):
            return False
        if pt.visibility is not None and pt.visibility < min_vis:
            return False
        if pt.x < 0.02 or pt.x > 0.98 or pt.y < 0.02 or pt.y > 0.96:
            return False
        return True

    def dist_2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def apply_deadzone(val: float, threshold: float = 0.035) -> float:
        if not math.isfinite(val) or abs(val) < threshold:
            return 0.0
        return val - threshold if val > 0 else val + threshold

    nose = pts[0]
    left_eye = pts[2]
    right_eye = pts[5]
    left_ear = pts[7]
    right_ear = pts[8]
    mouth_left = pts[9]
    mouth_right = pts[10]
    shoulder_left = pts[11]
    shoulder_right = pts[12]
    elbow_left = pts[13]
    elbow_right = pts[14]
    wrist_left = pts[15]
    wrist_right = pts[16]

    if (
        not is_reliable(nose, 0.5)
        or not is_reliable(shoulder_left, 0.5)
        or not is_reliable(shoulder_right, 0.5)
    ):
        return [0.0] * NMM_FEATURE_DIM

    shoulder_width = max(dist_2d((shoulder_left.x, shoulder_left.y), (shoulder_right.x, shoulder_right.y)), 0.22)
    shoulder_mid = ((shoulder_left.x + shoulder_right.x) / 2.0, (shoulder_left.y + shoulder_right.y) / 2.0)

    eyes_vis = is_reliable(left_eye, 0.55) and is_reliable(right_eye, 0.55)
    ears_vis = is_reliable(left_ear, 0.55) and is_reliable(right_ear, 0.55)
    mouth_vis = is_reliable(mouth_left, 0.55) and is_reliable(mouth_right, 0.55)

    eye_dx = (right_eye.x - left_eye.x) if eyes_vis else (shoulder_right.x - shoulder_left.x)
    eye_dy = (right_eye.y - left_eye.y) if eyes_vis else 0.0
    head_roll = apply_deadzone(math.atan2(eye_dy, max(abs(eye_dx), 0.05)), 0.05)

    head_to_shoulder_dist = (shoulder_mid[1] - nose.y) / shoulder_width
    head_pitch = apply_deadzone(head_to_shoulder_dist - 0.72, 0.05)

    head_yaw = 0.0
    if ears_vis:
        d_left = dist_2d((nose.x, nose.y), (left_ear.x, left_ear.y))
        d_right = dist_2d((nose.x, nose.y), (right_ear.x, right_ear.y))
        head_yaw = apply_deadzone((d_left - d_right) / (d_left + d_right + 1e-4), 0.06)

    mouth_mid = (
        ((mouth_left.x + mouth_right.x) / 2.0, (mouth_left.y + mouth_right.y) / 2.0)
        if mouth_vis
        else (nose.x, nose.y + 0.12 * shoulder_width)
    )
    shoulder_tilt = apply_deadzone((shoulder_right.y - shoulder_left.y) / shoulder_width, 0.04)

    has_le = is_reliable(elbow_left, 0.58)
    has_re = is_reliable(elbow_right, 0.58)
    has_lw = is_reliable(wrist_left, 0.58)
    has_rw = is_reliable(wrist_right, 0.58)

    le_sx = (elbow_left.x - shoulder_left.x) / shoulder_width if has_le else 0.0
    le_sy = (elbow_left.y - shoulder_left.y) / shoulder_width - 0.45 if has_le else 0.0
    re_sx = (elbow_right.x - shoulder_right.x) / shoulder_width if has_re else 0.0
    re_sy = (elbow_right.y - shoulder_right.y) / shoulder_width - 0.45 if has_re else 0.0

    lw_nx = (wrist_left.x - nose.x) / shoulder_width if has_lw else 0.0
    lw_ny = (wrist_left.y - nose.y) / shoulder_width if has_lw else 0.0
    lw_dn = dist_2d((wrist_left.x, wrist_left.y), (nose.x, nose.y)) / shoulder_width if has_lw else 0.0
    lw_dm = dist_2d((wrist_left.x, wrist_left.y), mouth_mid) / shoulder_width if has_lw else 0.0
    lw_cx = (wrist_left.x - shoulder_mid[0]) / shoulder_width if has_lw else 0.0
    lw_cy = (wrist_left.y - shoulder_mid[1]) / shoulder_width if has_lw else 0.0
    lw_dc = dist_2d((wrist_left.x, wrist_left.y), shoulder_mid) / shoulder_width if has_lw else 0.0

    rw_nx = (wrist_right.x - nose.x) / shoulder_width if has_rw else 0.0
    rw_ny = (wrist_right.y - nose.y) / shoulder_width if has_rw else 0.0
    rw_dn = dist_2d((wrist_right.x, wrist_right.y), (nose.x, nose.y)) / shoulder_width if has_rw else 0.0
    rw_dm = dist_2d((wrist_right.x, wrist_right.y), mouth_mid) / shoulder_width if has_rw else 0.0
    rw_cx = (wrist_right.x - shoulder_mid[0]) / shoulder_width if has_rw else 0.0
    rw_cy = (wrist_right.y - shoulder_mid[1]) / shoulder_width if has_rw else 0.0
    rw_dc = dist_2d((wrist_right.x, wrist_right.y), shoulder_mid) / shoulder_width if has_rw else 0.0

    active_wrist = wrist_right if has_rw else (wrist_left if has_lw else None)
    active_zone = (shoulder_mid[1] - active_wrist.y) / shoulder_width if active_wrist else 0.0
    active_d_head = dist_2d((active_wrist.x, active_wrist.y), (nose.x, nose.y)) / shoulder_width if active_wrist else 0.0
    active_d_chest = dist_2d((active_wrist.x, active_wrist.y), shoulder_mid) / shoulder_width if active_wrist else 0.0
    two_hands_dist = dist_2d((wrist_left.x, wrist_left.y), (wrist_right.x, wrist_right.y)) / shoulder_width if (has_lw and has_rw) else 0.0

    lf_x = (wrist_left.x - elbow_left.x) / shoulder_width if (has_le and has_lw) else 0.0
    lf_y = (wrist_left.y - elbow_left.y) / shoulder_width if (has_le and has_lw) else 0.0
    rf_x = (wrist_right.x - elbow_right.x) / shoulder_width if (has_re and has_rw) else 0.0
    rf_y = (wrist_right.y - elbow_right.y) / shoulder_width if (has_re and has_rw) else 0.0

    raw_vec = [
        head_roll, head_pitch, head_yaw, shoulder_tilt,
        le_sx, le_sy, re_sx, re_sy,
        lw_nx, lw_ny, lw_dn, lw_dm, lw_cx, lw_cy, lw_dc,
        rw_nx, rw_ny, rw_dn, rw_dm, rw_cx, rw_cy, rw_dc,
        active_zone, active_d_head, active_d_chest, two_hands_dist,
        lf_x, lf_y, rf_x, rf_y,
        0.5 if has_lw else 0.0,
        0.5 if has_rw else 0.0,
    ]

    out: List[float] = []
    for v in raw_vec[:NMM_FEATURE_DIM]:
        if not math.isfinite(v):
            out.append(0.0)
        else:
            out.append(round(max(-0.15, min(0.15, v * 0.1)), 6))
    return out


def extract_invariant_feature_vector(
    landmarks: Sequence[Union[Landmark3D, Dict[str, Any], Sequence[float]]],
    velocity: float = 0.0,
    delta_x: float = 0.0,
    delta_y: float = 0.0,
    delta_z: float = 0.0,
    pose_or_nmm: Optional[Sequence[Any]] = None,
    include_nmm: bool = True,
) -> List[float]:
    """
    Exact reproduction of `extractInvariantFeatureVector` in `src/utils/vslClassifier.ts`
    and `extract_invariant_feature_vector_py` in `verify_parity.py`.

    Returns:
      - 138-D vector when `include_nmm=True` (106 hand + 32 pose/NMM).
      - 106-D vector when `include_nmm=False` (106 canonical hand features).

    Layout of the first 106 dimensions:
      [0..14]   15 Joint flexion angles * 2.5
      [15..19]   5 Finger extension ratios * (3.0 for thumb, 3.5 for others)
      [20..29]  10 Inter-fingertip distances / palmScale * weight (3.5 or 2.0)
      [30..34]   5 Fingertip-to-palm-center distances / palmScale * 2.5
      [35..38]   4 Thumb-to-MCP distances / palmScale * 2.5
      [39..101] 63 Rotation-aligned 3D coordinates * 0.8
      [102..105] 4 Wrist motion dynamics: [delta_x*6.0, delta_y*6.0, delta_z*6.0, velocity*8.0]
    """
    target_dim = SINGLE_SIGN_FEATURE_DIM if include_nmm else SINGLE_HAND_FEATURE_DIM
    if not landmarks or len(landmarks) < 21:
        return [0.0] * target_dim

    pts = [Landmark3D.from_any(pt) for pt in landmarks[:21]]

    # Sanitize input coordinates against NaN/Inf before geometry calculations
    for pt in pts:
        pt.x = sanitize_float(pt.x, 0.0)
        pt.y = sanitize_float(pt.y, 0.0)
        pt.z = sanitize_float(pt.z, 0.0)

    palm_scale = max(distance_3d(pts[0], pts[9]), 0.5)
    palm_center = Landmark3D(
        x=(pts[0].x + pts[5].x + pts[9].x + pts[17].x) / 4.0,
        y=(pts[0].y + pts[5].y + pts[9].y + pts[17].y) / 4.0,
        z=(pts[0].z + pts[5].z + pts[9].z + pts[17].z) / 4.0,
    )

    vec: List[float] = []

    # 1. 15 Joint Angles (radians * 2.5)
    triplets = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),       # Thumb
        (0, 5, 6), (5, 6, 7), (6, 7, 8),       # Index
        (0, 9, 10), (9, 10, 11), (10, 11, 12), # Middle
        (0, 13, 14), (13, 14, 15), (14, 15, 16), # Ring
        (0, 17, 18), (17, 18, 19), (18, 19, 20), # Pinky
    ]
    for a_idx, b_idx, c_idx in triplets:
        vec.append(calculate_angle_3d(pts[a_idx], pts[b_idx], pts[c_idx]) * 2.5)

    # 2. 5 Finger Extension Ratios
    vec.append((distance_3d(pts[4], pts[17]) / palm_scale) * 3.0)
    vec.append((distance_3d(pts[8], pts[0]) / palm_scale) * 3.5)
    vec.append((distance_3d(pts[12], pts[0]) / palm_scale) * 3.5)
    vec.append((distance_3d(pts[16], pts[0]) / palm_scale) * 3.5)
    vec.append((distance_3d(pts[20], pts[0]) / palm_scale) * 3.5)

    # 3. 10 Inter-Fingertip Distances
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            d = distance_3d(pts[tips[i]], pts[tips[j]]) / palm_scale
            weight = 3.5 if ((i == 0 and j == 1) or (i == 1 and j == 2)) else 2.0
            vec.append(d * weight)

    # 4. 5 Fingertip-to-Palm-Center Distances
    for tip_idx in tips:
        vec.append((distance_3d(pts[tip_idx], palm_center) / palm_scale) * 2.5)

    # 5. 4 Thumb-to-MCP Distances
    for mcp_idx in (5, 9, 13, 17):
        vec.append((distance_3d(pts[4], pts[mcp_idx]) / palm_scale) * 2.5)

    # 6. 63 Rotation-Aligned 3D Coordinates
    align_angle = math.atan2(pts[9].y - pts[0].y, pts[9].x - pts[0].x)
    cos_a = math.cos(-align_angle)
    sin_a = math.sin(-align_angle)
    for pt in pts:
        rx = pt.x * cos_a - pt.y * sin_a
        ry = pt.x * sin_a + pt.y * cos_a
        vec.extend([rx * 0.8, ry * 0.8, pt.z * 0.8])

    # 7. 4 Wrist Motion Dynamics
    dx = sanitize_float(delta_x, 0.0)
    dy = sanitize_float(delta_y, 0.0)
    dz = sanitize_float(delta_z, 0.0)
    vel = sanitize_float(velocity, 0.0)
    vec.extend([dx * 6.0, dy * 6.0, dz * 6.0, vel * 8.0])

    if not include_nmm:
        return sanitize_vector(vec[:SINGLE_HAND_FEATURE_DIM], SINGLE_HAND_FEATURE_DIM)

    # 8. 32 Pose / NMM Features
    if pose_or_nmm is not None and len(pose_or_nmm) > 0 and isinstance(pose_or_nmm[0], (int, float)):
        for i in range(NMM_FEATURE_DIM):
            val = sanitize_float(pose_or_nmm[i] if i < len(pose_or_nmm) else 0.0, 0.0)
            vec.append(max(-0.15, min(0.15, val)))
    else:
        nmm_vec = extract_nmm_feature_vector(pose_or_nmm)  # type: ignore[arg-type]
        vec.extend(nmm_vec)

    return sanitize_vector(vec, SINGLE_SIGN_FEATURE_DIM)


def extract_single_hand_106(
    landmarks: Sequence[Union[Landmark3D, Dict[str, Any], Sequence[float]]],
    velocity: float = 0.0,
    delta_x: float = 0.0,
    delta_y: float = 0.0,
    delta_z: float = 0.0,
) -> List[float]:
    """Convenience wrapper returning the exact 106-D single-hand feature vector."""
    return extract_invariant_feature_vector(
        landmarks=landmarks,
        velocity=velocity,
        delta_x=delta_x,
        delta_y=delta_y,
        delta_z=delta_z,
        include_nmm=False,
    )


# =============================================================================
# 3. CANONICAL 212-D TWO-HAND FEATURE EXTRACTION (LEFT 106 + RIGHT 106)
# =============================================================================

@dataclass
class HandTrackState:
    prev_wrist_left: Optional[Landmark3D] = None
    prev_wrist_right: Optional[Landmark3D] = None


def create_hand_track_state() -> HandTrackState:
    return HandTrackState()


@dataclass
class Canonical212FrameResult:
    combined212: List[float]
    left106: List[float]
    right106: List[float]
    has_left: bool
    has_right: bool
    is_occluded: bool
    hand_count: int


def _resolve_handedness_label(raw_item: Any) -> str:
    """Resolve MediaPipe or Vietnamese handedness metadata into 'Left', 'Right', or 'Unknown'."""
    if raw_item is None:
        return "Unknown"
    if isinstance(raw_item, (list, tuple)) and len(raw_item) > 0:
        raw_item = raw_item[0]
    if isinstance(raw_item, dict):
        name = raw_item.get("categoryName") or raw_item.get("displayName") or raw_item.get("handedness") or ""
    elif isinstance(raw_item, str):
        name = raw_item
    else:
        name = getattr(raw_item, "categoryName", None) or getattr(raw_item, "displayName", "")

    name_clean = str(name).strip()
    if name_clean in ("Left", "Tay trái", "left", "LEFT"):
        return "Left"
    if name_clean in ("Right", "Tay phải", "right", "RIGHT"):
        return "Right"
    return "Unknown"


def extract_canonical_212_frame(
    screen_hands: Sequence[Sequence[Union[Landmark3D, Dict[str, Any], Sequence[float]]]],
    raw_handedness: Optional[Sequence[Any]] = None,
    tracker: Optional[HandTrackState] = None,
) -> Canonical212FrameResult:
    """
    Exact reproduction of `extractCanonical212Frame` in `src/utils/ctcFeatureExtractor.ts`:
    - First 106 dimensions: Canonical LEFT hand (`left106`).
    - Last 106 dimensions: Canonical RIGHT hand (`right106`).
    - Missing hand slot is zero-padded with 106 zeros and its wrist velocity tracker is reset.
    - Hand assignment prioritizes MediaPipe handedness ('Left'/'Right') and falls back to
      deterministic spatial wristX ordering (wristX < 0.5 or handA.wristX <= handB.wristX -> Left).
    """
    if tracker is None:
        tracker = create_hand_track_state()

    zero_106 = [0.0] * SINGLE_HAND_FEATURE_DIM

    if not screen_hands or len(screen_hands) == 0:
        tracker.prev_wrist_left = None
        tracker.prev_wrist_right = None
        return Canonical212FrameResult(
            combined212=[0.0] * CTC_FEATURE_DIM,
            left106=list(zero_106),
            right106=list(zero_106),
            has_left=False,
            has_right=False,
            is_occluded=True,
            hand_count=0,
        )

    classified: List[Dict[str, Any]] = []
    for idx, hand_lm in enumerate(screen_hands):
        pts = [Landmark3D.from_any(p) for p in hand_lm]
        h_label = "Unknown"
        if raw_handedness and idx < len(raw_handedness):
            h_label = _resolve_handedness_label(raw_handedness[idx])
        wrist_x = pts[0].x if len(pts) > 0 else 0.5
        classified.append({
            "landmarks": pts,
            "handedness": h_label,
            "wrist_x": wrist_x,
        })

    left_lm: Optional[List[Landmark3D]] = None
    right_lm: Optional[List[Landmark3D]] = None

    if len(classified) == 1:
        single = classified[0]
        if single["handedness"] == "Left":
            left_lm = single["landmarks"]
        elif single["handedness"] == "Right":
            right_lm = single["landmarks"]
        else:
            if single["wrist_x"] < 0.5:
                left_lm = single["landmarks"]
            else:
                right_lm = single["landmarks"]
    else:
        hand_a = classified[0]
        hand_b = classified[1]

        if hand_a["handedness"] == "Left" and hand_b["handedness"] == "Right":
            left_lm = hand_a["landmarks"]
            right_lm = hand_b["landmarks"]
        elif hand_a["handedness"] == "Right" and hand_b["handedness"] == "Left":
            left_lm = hand_b["landmarks"]
            right_lm = hand_a["landmarks"]
        elif hand_a["handedness"] == "Left" and hand_b["handedness"] != "Left":
            left_lm = hand_a["landmarks"]
            right_lm = hand_b["landmarks"]
        elif hand_a["handedness"] == "Right" and hand_b["handedness"] != "Right":
            left_lm = hand_b["landmarks"]
            right_lm = hand_a["landmarks"]
        elif hand_b["handedness"] == "Left" and hand_a["handedness"] != "Left":
            left_lm = hand_b["landmarks"]
            right_lm = hand_a["landmarks"]
        elif hand_b["handedness"] == "Right" and hand_a["handedness"] != "Right":
            left_lm = hand_a["landmarks"]
            right_lm = hand_b["landmarks"]
        else:
            if hand_a["wrist_x"] <= hand_b["wrist_x"]:
                left_lm = hand_a["landmarks"]
                right_lm = hand_b["landmarks"]
            else:
                left_lm = hand_b["landmarks"]
                right_lm = hand_a["landmarks"]

    # Extract Left 106-D
    if left_lm is not None and len(left_lm) == 21:
        norm_left, wrist_left, _ = normalize_landmarks(left_lm)
        dx, dy, dz, vel = 0.0, 0.0, 0.0, 0.0
        if tracker.prev_wrist_left is not None:
            dx = wrist_left.x - tracker.prev_wrist_left.x
            dy = wrist_left.y - tracker.prev_wrist_left.y
            dz = (wrist_left.z or 0.0) - (tracker.prev_wrist_left.z or 0.0)
            vel = math.sqrt(dx * dx + dy * dy + dz * dz)
        tracker.prev_wrist_left = Landmark3D(wrist_left.x, wrist_left.y, wrist_left.z)
        left106 = extract_single_hand_106(norm_left, velocity=vel, delta_x=dx, delta_y=dy, delta_z=dz)
    else:
        tracker.prev_wrist_left = None
        left106 = list(zero_106)
        left_lm = None

    # Extract Right 106-D
    if right_lm is not None and len(right_lm) == 21:
        norm_right, wrist_right, _ = normalize_landmarks(right_lm)
        dx, dy, dz, vel = 0.0, 0.0, 0.0, 0.0
        if tracker.prev_wrist_right is not None:
            dx = wrist_right.x - tracker.prev_wrist_right.x
            dy = wrist_right.y - tracker.prev_wrist_right.y
            dz = (wrist_right.z or 0.0) - (tracker.prev_wrist_right.z or 0.0)
            vel = math.sqrt(dx * dx + dy * dy + dz * dz)
        tracker.prev_wrist_right = Landmark3D(wrist_right.x, wrist_right.y, wrist_right.z)
        right106 = extract_single_hand_106(norm_right, velocity=vel, delta_x=dx, delta_y=dy, delta_z=dz)
    else:
        tracker.prev_wrist_right = None
        right106 = list(zero_106)
        right_lm = None

    combined212 = left106 + right106
    return Canonical212FrameResult(
        combined212=combined212,
        left106=left106,
        right106=right106,
        has_left=left_lm is not None,
        has_right=right_lm is not None,
        is_occluded=False,
        hand_count=len(screen_hands),
    )


def interpolate_occluded_frames(
    raw_frames: Sequence[Dict[str, Any]],
    max_gap_ms: float = MAX_OCCLUSION_INTERPOLATION_GAP_MS,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Exact reproduction of `interpolateOccludedFrames` in `src/components/ContinuousSentenceRecorder.tsx`:
    Linearly interpolates short internal occlusion gaps (<= max_gap_ms) bounded by valid frames
    on both sides. Leading and trailing occlusion segments are never interpolated.
    """
    frames: List[Dict[str, Any]] = []
    for f in raw_frames:
        f_copy = dict(f)
        if "featureVector" in f_copy and f_copy["featureVector"] is not None:
            f_copy["featureVector"] = list(f_copy["featureVector"])
        if "featureVectorHand2" in f_copy and f_copy["featureVectorHand2"] is not None:
            f_copy["featureVectorHand2"] = list(f_copy["featureVectorHand2"])
        if "combinedFeatureVector" in f_copy and f_copy["combinedFeatureVector"] is not None:
            f_copy["combinedFeatureVector"] = list(f_copy["combinedFeatureVector"])
        frames.append(f_copy)

    occluded_count = 0
    interpolated_count = 0

    for f in frames:
        if f.get("isOccluded") or f.get("handCount", 1) == 0:
            f["isOccluded"] = True
            occluded_count += 1
        else:
            f["isOccluded"] = False
            f["isInterpolated"] = False

    i = 0
    n = len(frames)
    while i < n:
        if frames[i].get("isOccluded"):
            start_idx = i
            while i < n and frames[i].get("isOccluded"):
                i += 1
            end_idx = i - 1

            prev_valid_idx = start_idx - 1
            next_valid_idx = end_idx + 1

            if prev_valid_idx >= 0 and next_valid_idx < n:
                t_prev = float(frames[prev_valid_idx].get("timestamp", prev_valid_idx * 33.333))
                t_next = float(frames[next_valid_idx].get("timestamp", next_valid_idx * 33.333))
                gap_ms = t_next - t_prev

                if 0 < gap_ms <= max_gap_ms:
                    prev_f = frames[prev_valid_idx]
                    next_f = frames[next_valid_idx]

                    prev_comb = prev_f.get("combinedFeatureVector")
                    next_comb = next_f.get("combinedFeatureVector")

                    for k in range(start_idx, end_idx + 1):
                        curr_f = frames[k]
                        t_curr = float(curr_f.get("timestamp", k * 33.333))
                        alpha = (t_curr - t_prev) / gap_ms

                        if prev_comb and next_comb and len(prev_comb) == CTC_FEATURE_DIM and len(next_comb) == CTC_FEATURE_DIM:
                            interp_212 = [
                                prev_comb[d] + alpha * (next_comb[d] - prev_comb[d])
                                for d in range(CTC_FEATURE_DIM)
                            ]
                            curr_f["combinedFeatureVector"] = interp_212
                            curr_f["featureVector"] = interp_212[:SINGLE_HAND_FEATURE_DIM]
                            curr_f["featureVectorHand2"] = interp_212[SINGLE_HAND_FEATURE_DIM:]
                        else:
                            v1_prev = prev_f.get("featureVector") or [0.0] * SINGLE_HAND_FEATURE_DIM
                            v1_next = next_f.get("featureVector") or [0.0] * SINGLE_HAND_FEATURE_DIM
                            new_vec1 = [
                                v1_prev[d] + alpha * (v1_next[d] - v1_prev[d])
                                for d in range(SINGLE_HAND_FEATURE_DIM)
                            ]
                            curr_f["featureVector"] = new_vec1

                            v2_prev = prev_f.get("featureVectorHand2")
                            v2_next = next_f.get("featureVectorHand2")
                            if v2_prev or v2_next:
                                v2_p = v2_prev or [0.0] * SINGLE_HAND_FEATURE_DIM
                                v2_n = v2_next or [0.0] * SINGLE_HAND_FEATURE_DIM
                                new_vec2 = [
                                    v2_p[d] + alpha * (v2_n[d] - v2_p[d])
                                    for d in range(SINGLE_HAND_FEATURE_DIM)
                                ]
                            else:
                                new_vec2 = [0.0] * SINGLE_HAND_FEATURE_DIM
                            curr_f["featureVectorHand2"] = new_vec2
                            curr_f["combinedFeatureVector"] = new_vec1 + new_vec2

                        curr_f["isInterpolated"] = True
                        curr_f["isOccluded"] = True
                        interpolated_count += 1
        else:
            i += 1

    return frames, occluded_count, interpolated_count


# =============================================================================
# 4. SINGLE-SIGN SEQUENCE RESAMPLING & IDLE BOUNDARY TRIMMING (30 FRAMES)
# =============================================================================

def resample_sequence(
    raw_vecs: Sequence[Sequence[float]],
    target_len: int = SINGLE_SIGN_SEQUENCE_LENGTH,
    expected_dim: int = SINGLE_SIGN_FEATURE_DIM,
) -> List[List[float]]:
    """
    Exact reproduction of `resampleSequence` in `src/utils/lstmModel.ts`:
    Uniform temporal linear interpolation to `target_len` frames (default 30).
    """
    if not raw_vecs or len(raw_vecs) == 0:
        return [[0.0] * expected_dim for _ in range(target_len)]

    num_features = len(raw_vecs[0])
    if num_features != expected_dim:
        raise ValueError(f"resample_sequence feature dimension mismatch: expected {expected_dim}, got {num_features}")

    for idx, frame in enumerate(raw_vecs):
        if len(frame) != num_features:
            raise ValueError(f"Inconsistent feature vector length at frame {idx}: expected {num_features}, got {len(frame)}")

    orig_len = len(raw_vecs)
    if orig_len == target_len:
        return [list(row) for row in raw_vecs]

    resampled: List[List[float]] = []
    for i in range(target_len):
        raw_pos = (i / (target_len - 1)) * (orig_len - 1) if target_len > 1 else 0.0
        low_idx = int(math.floor(raw_pos))
        high_idx = min(low_idx + 1, orig_len - 1)
        alpha = raw_pos - low_idx

        frame_vec = [
            raw_vecs[low_idx][f] + alpha * (raw_vecs[high_idx][f] - raw_vecs[low_idx][f])
            for f in range(num_features)
        ]
        resampled.append(frame_vec)

    return resampled


def trim_sequence_idle_boundaries(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Exact reproduction of `trimSequenceIdleBoundaries` in `src/utils/lstmModel.ts`:
    Trims dead idle frames at the start and end of a fixed recording window while keeping
    2 padding frames before and 2 settling frames after active motion.
    """
    row_list = list(rows)
    if len(row_list) < 10:
        return row_list

    tip_indices = (4, 8, 12, 16, 20)
    typical_palm_scale = 0.15
    motions = [0.0] * len(row_list)

    for i, curr in enumerate(row_list):
        dx_w = sanitize_float(curr.get("delta_x", 0.0))
        dy_w = sanitize_float(curr.get("delta_y", 0.0))
        dz_w = sanitize_float(curr.get("delta_z", 0.0))
        rec_vel = sanitize_float(curr.get("velocity", 0.0))
        wrist_v = rec_vel if rec_vel > 0 else math.sqrt(dx_w * dx_w + dy_w * dy_w + dz_w * dz_w)

        if i == 0:
            motions[i] = wrist_v
            continue

        prev = row_list[i - 1]
        max_tip_v = 0.0
        for idx in tip_indices:
            dx_t = dx_w + (sanitize_float(curr.get(f"x_{idx}", 0.0)) - sanitize_float(prev.get(f"x_{idx}", 0.0))) * typical_palm_scale
            dy_t = dy_w + (sanitize_float(curr.get(f"y_{idx}", 0.0)) - sanitize_float(prev.get(f"y_{idx}", 0.0))) * typical_palm_scale
            dz_t = dz_w + (sanitize_float(curr.get(f"z_{idx}", 0.0)) - sanitize_float(prev.get(f"z_{idx}", 0.0))) * typical_palm_scale
            v_t = math.sqrt(dx_t * dx_t + dy_t * dy_t + dz_t * dz_t)
            if v_t > max_tip_v:
                max_tip_v = v_t

        motions[i] = max(wrist_v, 0.4 * wrist_v + 0.6 * max_tip_v)

    peak_motion = max(motions) if motions else 0.0
    if peak_motion < 0.018:
        return row_list

    active_threshold = max(0.008, min(0.022, peak_motion * 0.2))
    first_active = -1
    last_active = -1

    for i in range(1, len(row_list)):
        if motions[i] >= active_threshold:
            if first_active == -1:
                first_active = i
            last_active = i

    if first_active == -1 or last_active < first_active:
        return row_list

    start_idx = max(0, first_active - 2)
    end_idx = min(len(row_list) - 1, last_active + 2)

    if end_idx - start_idx + 1 < 6:
        return row_list

    return row_list[start_idx : end_idx + 1]


# =============================================================================
# 5. CONTINUOUS SIGN BOUNDARY DETECTOR (PYTHON PORT OF signBoundaryDetector.ts)
# =============================================================================

@dataclass
class BoundaryDetectionConfig:
    velocity_threshold: float = 0.024
    min_rest_ms: float = 160.0
    min_motion_ms: float = 180.0
    min_cumulative_motion: float = 0.08
    min_total_displacement: float = 0.04
    max_lost_tolerance_ms: float = 200.0
    sliding_window_capacity: int = 200
    pre_motion_padding_ms: float = 100.0
    pause_bridging_grace_ms: float = 220.0


@dataclass
class FrameSample:
    timestamp: float
    frame_index: int
    feature_vector: List[float]
    wrist_velocity: float
    fingertip_velocity: float
    composite_velocity: float
    delta_x: float
    delta_y: float
    delta_z: float
    is_moving: bool
    is_interpolated: bool
    hand_count: int


@dataclass
class SignSegment:
    id: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    duration_ms: int
    frame_count: int
    frames: List[List[float]]
    raw_samples: List[FrameSample]
    peak_velocity: float
    avg_velocity: float
    cumulative_motion: float
    total_displacement: float
    interpolated_frame_count: int
    interpolated_ratio: float
    status: str = "completed"


@dataclass
class RejectedSegmentInfo:
    id: str
    timestamp: float
    frame_count: int
    valid_frame_count: int
    duration_ms: int
    reason: str


@dataclass
class _TrackedHandState:
    label: Optional[str]
    wrist: Landmark3D
    tips: List[Landmark3D]
    last_seen_timestamp: float


class SignBoundaryDetector:
    """
    Python implementation of `SignBoundaryDetector` from `src/utils/signBoundaryDetector.ts`.
    Preserves:
      - 3-Phase State Machine: REST <-> MOTION <-> BRIDGING_PAUSE
      - Composite velocity: 0.4 * wrist_velocity + 0.6 * max_fingertip_velocity (FPS-normalized)
      - Hand identity tracking (Handedness priority + Bipartite wrist distance fallback with 0.35 jump rejection)
      - Adaptive EMA noise floor
      - Landmark dropout tolerance (max_lost_tolerance_ms = 200ms)
      - Brief-pause bridging (pause_bridging_grace_ms = 220ms)
      - Pre-motion padding with boundary contamination protection
    """

    def __init__(self, config: Optional[BoundaryDetectionConfig] = None) -> None:
        self.config = config or BoundaryDetectionConfig()
        self.sliding_window_buffer: List[FrameSample] = []
        self.current_phase: str = "REST"
        self.consecutive_rest_frames: int = 0
        self.consecutive_motion_frames: int = 0
        self.bridging_pause_frames: int = 0
        self.consecutive_lost_frames: int = 0
        self.active_segment_frames: List[FrameSample] = []
        self.completed_segments_buffer: List[SignSegment] = []
        self.frame_counter: int = 0
        self.active_segment_start_time: float = 0.0
        self.active_segment_start_frame: int = 0

        self.motion_start_time: float = 0.0
        self.bridging_pause_start_time: float = 0.0
        self.lost_start_time: float = 0.0

        self.prev_hands_by_label: Dict[str, _TrackedHandState] = {}
        self.prev_hands_fallback: List[_TrackedHandState] = []
        self.last_timestamp: float = 0.0
        self.last_boundary_timestamp: float = -1.0
        self.last_boundary_frame_index: int = -1

        self.noise_floor_ema: float = 0.012
        self.last_rejected_segment: Optional[RejectedSegmentInfo] = None

    def _compute_single_hand_velocity(
        self,
        landmarks: Sequence[Landmark3D],
        prev_data: Optional[_TrackedHandState],
        time_scale: float = 1.0,
    ) -> Tuple[float, float, float, List[Landmark3D]]:
        if not landmarks or len(landmarks) < 21:
            return 0.0, 0.0, 0.0, []
        wrist = landmarks[0]
        tip_indices = (4, 8, 12, 16, 20)
        current_tips = [landmarks[idx] for idx in tip_indices]
        if prev_data is None:
            return 0.0, 0.0, 0.0, current_tips

        wrist_vel = distance_3d(wrist, prev_data.wrist) * time_scale
        max_tip_vel = 0.0
        for cur_t, prev_t in zip(current_tips, prev_data.tips):
            v_t = distance_3d(cur_t, prev_t) * time_scale
            if v_t > max_tip_vel:
                max_tip_vel = v_t

        comp_vel = 0.4 * wrist_vel + 0.6 * max_tip_vel
        return wrist_vel, max_tip_vel, comp_vel, current_tips

    def _match_and_compute_hands_velocity(
        self,
        hands: Sequence[Sequence[Landmark3D]],
        time_scale: float,
        timestamp: float,
        hand_labels: Optional[Sequence[Optional[str]]] = None,
    ) -> Tuple[float, float, float]:
        tip_indices = (4, 8, 12, 16, 20)

        def make_state(lm: Sequence[Landmark3D], lbl: Optional[str]) -> _TrackedHandState:
            return _TrackedHandState(
                label=lbl if lbl in ("Left", "Right") else None,
                wrist=lm[0],
                tips=[lm[idx] for idx in tip_indices],
                last_seen_timestamp=timestamp,
            )

        if len(hands) == 0:
            self.prev_hands_by_label = {}
            self.prev_hands_fallback = []
            return 0.0, 0.0, 0.0

        current_states = [
            make_state(lm, hand_labels[idx] if (hand_labels and idx < len(hand_labels)) else None)
            for idx, lm in enumerate(hands)
        ]

        labels_present = [s.label for s in current_states if s.label in ("Left", "Right")]
        has_valid_handedness = (
            len(labels_present) > 0
            and (len(labels_present) == 1 or (len(labels_present) == 2 and labels_present[0] != labels_present[1]))
        )

        if has_valid_handedness:
            max_w, max_t, max_c = 0.0, 0.0, 0.0
            next_by_label = dict(self.prev_hands_by_label)
            for curr_hand, curr_state in zip(hands, current_states):
                lbl = curr_state.label
                if lbl in ("Left", "Right"):
                    prev = self.prev_hands_by_label.get(lbl)
                    if prev is not None:
                        dist = distance_3d(curr_hand[0], prev.wrist)
                        if dist <= 0.35:
                            w_v, t_v, c_v, _ = self._compute_single_hand_velocity(curr_hand, prev, time_scale)
                            max_w = max(max_w, w_v)
                            max_t = max(max_t, t_v)
                            max_c = max(max_c, c_v)
                    next_by_label[lbl] = curr_state
            self.prev_hands_by_label = next_by_label
            self.prev_hands_fallback = current_states
            return max_w, max_t, max_c

        if len(self.prev_hands_fallback) == 0:
            self.prev_hands_fallback = current_states
            return 0.0, 0.0, 0.0

        if len(hands) == 1:
            curr = hands[0]
            curr_state = current_states[0]
            dists = [distance_3d(curr[0], p.wrist) for p in self.prev_hands_fallback]
            best_idx = 0 if (len(dists) == 1 or dists[0] <= dists[1]) else 1
            if dists[best_idx] > 0.35:
                self.prev_hands_fallback = [curr_state]
                return 0.0, 0.0, 0.0
            w_v, t_v, c_v, _ = self._compute_single_hand_velocity(curr, self.prev_hands_fallback[best_idx], time_scale)
            self.prev_hands_fallback = [curr_state]
            return w_v, t_v, c_v

        curr0, curr1 = hands[0], hands[1]
        state0, state1 = current_states[0], current_states[1]
        if len(self.prev_hands_fallback) == 1:
            d0 = distance_3d(curr0[0], self.prev_hands_fallback[0].wrist)
            d1 = distance_3d(curr1[0], self.prev_hands_fallback[0].wrist)
            res0 = (0.0, 0.0, 0.0)
            res1 = (0.0, 0.0, 0.0)
            if d0 <= d1 and d0 <= 0.35:
                res0 = self._compute_single_hand_velocity(curr0, self.prev_hands_fallback[0], time_scale)[:3]
            elif d1 < d0 and d1 <= 0.35:
                res1 = self._compute_single_hand_velocity(curr1, self.prev_hands_fallback[0], time_scale)[:3]
            self.prev_hands_fallback = [state0, state1]
            return max(res0[0], res1[0]), max(res0[1], res1[1]), max(res0[2], res1[2])

        cost_direct = (
            distance_3d(curr0[0], self.prev_hands_fallback[0].wrist)
            + distance_3d(curr1[0], self.prev_hands_fallback[1].wrist)
        )
        cost_swapped = (
            distance_3d(curr0[0], self.prev_hands_fallback[1].wrist)
            + distance_3d(curr1[0], self.prev_hands_fallback[0].wrist)
        )
        res0 = (0.0, 0.0, 0.0)
        res1 = (0.0, 0.0, 0.0)
        if cost_swapped < cost_direct:
            if distance_3d(curr0[0], self.prev_hands_fallback[1].wrist) <= 0.35:
                res0 = self._compute_single_hand_velocity(curr0, self.prev_hands_fallback[1], time_scale)[:3]
            if distance_3d(curr1[0], self.prev_hands_fallback[0].wrist) <= 0.35:
                res1 = self._compute_single_hand_velocity(curr1, self.prev_hands_fallback[0], time_scale)[:3]
        else:
            if distance_3d(curr0[0], self.prev_hands_fallback[0].wrist) <= 0.35:
                res0 = self._compute_single_hand_velocity(curr0, self.prev_hands_fallback[0], time_scale)[:3]
            if distance_3d(curr1[0], self.prev_hands_fallback[1].wrist) <= 0.35:
                res1 = self._compute_single_hand_velocity(curr1, self.prev_hands_fallback[1], time_scale)[:3]

        self.prev_hands_fallback = [state0, state1]
        return max(res0[0], res1[0]), max(res0[1], res1[1]), max(res0[2], res1[2])

    def _create_segment(self, samples: List[FrameSample], current_timestamp: float) -> SignSegment:
        velocities = [s.composite_velocity for s in samples]
        peak_vel = max(velocities) if velocities else 0.0
        avg_vel = sum(velocities) / max(len(velocities), 1)
        cum_motion = sum(velocities)
        total_disp = 0.0
        interp_count = 0

        for i, s in enumerate(samples):
            if s.is_interpolated:
                interp_count += 1
            if i > 0:
                total_disp += math.sqrt(s.delta_x ** 2 + s.delta_y ** 2 + s.delta_z ** 2)

        start_time = samples[0].timestamp if samples else current_timestamp
        end_time = samples[-1].timestamp if samples else current_timestamp
        duration_ms = max(1, int(round(end_time - start_time)))

        return SignSegment(
            id=f"seg_{int(start_time)}_{samples[0].frame_index if samples else 0}",
            start_time=start_time,
            end_time=end_time,
            start_frame=samples[0].frame_index if samples else 0,
            end_frame=samples[-1].frame_index if samples else 0,
            duration_ms=duration_ms,
            frame_count=len(samples),
            frames=[list(s.feature_vector) for s in samples],
            raw_samples=list(samples),
            peak_velocity=round(peak_vel, 4),
            avg_velocity=round(avg_vel, 4),
            cumulative_motion=round(cum_motion, 4),
            total_displacement=round(total_disp, 4),
            interpolated_frame_count=interp_count,
            interpolated_ratio=round(interp_count / max(len(samples), 1), 2),
            status="completed",
        )

    def _evaluate_and_emit_candidate(
        self,
        core_samples: List[FrameSample],
        timestamp: float,
        effective_threshold: float,
    ) -> Optional[SignSegment]:
        if not core_samples:
            return None
        duration_ms = (
            core_samples[-1].timestamp - core_samples[0].timestamp
            if len(core_samples) >= 2
            else len(core_samples) * 33.3
        )
        valid_count = sum(1 for s in core_samples if not s.is_interpolated and s.hand_count > 0)

        if duration_ms >= self.config.min_motion_ms:
            seg = self._create_segment(core_samples, timestamp)
            if (
                seg.cumulative_motion >= self.config.min_cumulative_motion
                or seg.total_displacement >= self.config.min_total_displacement
                or seg.peak_velocity >= effective_threshold * 1.35
                or seg.avg_velocity >= effective_threshold * 0.9
            ):
                self.completed_segments_buffer.append(seg)
                if len(self.completed_segments_buffer) > 100:
                    self.completed_segments_buffer.pop(0)
                return seg
            else:
                self.last_rejected_segment = RejectedSegmentInfo(
                    id=f"rej_{int(timestamp)}",
                    timestamp=timestamp,
                    frame_count=len(core_samples),
                    valid_frame_count=valid_count,
                    duration_ms=int(round(duration_ms)),
                    reason=f"Biên độ chuyển động quá thấp (Energy={seg.cumulative_motion:.2f} < {self.config.min_cumulative_motion})",
                )
        else:
            self.last_rejected_segment = RejectedSegmentInfo(
                id=f"rej_{int(timestamp)}",
                timestamp=timestamp,
                frame_count=len(core_samples),
                valid_frame_count=valid_count,
                duration_ms=int(round(duration_ms)),
                reason=f"Thời lượng cử chỉ quá ngắn ({int(round(duration_ms))}ms < {int(self.config.min_motion_ms)}ms, {len(core_samples)} frames)",
            )
        return None

    def process_frame(
        self,
        timestamp: float,
        feature_vector: Sequence[float],
        hands_landmarks: Optional[Sequence[Sequence[Union[Landmark3D, Dict[str, Any], Sequence[float]]]]] = None,
        hand_labels: Optional[Sequence[Optional[str]]] = None,
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        delta_z: float = 0.0,
        fallback_velocity: Optional[float] = None,
    ) -> Optional[SignSegment]:
        self.frame_counter += 1
        hands_pts: List[List[Landmark3D]] = []
        if hands_landmarks is not None:
            hands_pts = [[Landmark3D.from_any(p) for p in h] for h in hands_landmarks if len(h) >= 21]
        hand_count = len(hands_pts) if hands_landmarks is not None else (1 if fallback_velocity is not None else 0)

        time_scale = 1.0
        if self.last_timestamp > 0 and timestamp > self.last_timestamp:
            dt = timestamp - self.last_timestamp
            if 5.0 <= dt <= 250.0:
                time_scale = 33.3333 / dt
        self.last_timestamp = timestamp

        # 1. Handle temporary landmark loss
        if hand_count == 0:
            if self.lost_start_time == 0.0:
                self.lost_start_time = timestamp
            self.consecutive_lost_frames += 1
            lost_duration_ms = max(0.0, timestamp - self.lost_start_time)

            if lost_duration_ms <= self.config.max_lost_tolerance_ms:
                if self.current_phase in ("MOTION", "BRIDGING_PAUSE") and len(self.active_segment_frames) > 0:
                    last_sample = self.active_segment_frames[-1]
                    self.active_segment_frames.append(
                        FrameSample(
                            timestamp=timestamp,
                            frame_index=self.frame_counter,
                            feature_vector=list(last_sample.feature_vector),
                            wrist_velocity=0.0,
                            fingertip_velocity=0.0,
                            composite_velocity=0.005,
                            delta_x=0.0,
                            delta_y=0.0,
                            delta_z=0.0,
                            is_moving=False,
                            is_interpolated=True,
                            hand_count=0,
                        )
                    )
                return None
            else:
                seg_on_exit: Optional[SignSegment] = None
                if self.current_phase in ("MOTION", "BRIDGING_PAUSE") and len(self.active_segment_frames) > 0:
                    cleaned = list(self.active_segment_frames)
                    while cleaned and cleaned[-1].is_interpolated:
                        cleaned.pop()
                    if self.current_phase == "BRIDGING_PAUSE" and self.bridging_pause_frames > 2:
                        trim_cnt = min(self.bridging_pause_frames - 2, max(0, len(cleaned) - 1))
                        if trim_cnt > 0:
                            cleaned = cleaned[:-trim_cnt]
                    exit_ts = cleaned[-1].timestamp if cleaned else timestamp
                    seg_on_exit = self._evaluate_and_emit_candidate(
                        cleaned, exit_ts, self.config.velocity_threshold
                    )
                self.reset(preserve_last_rejected=True)
                return seg_on_exit

        self.consecutive_lost_frames = 0
        self.lost_start_time = 0.0

        # 2. Compute composite velocity with hand identity tracking
        if hands_landmarks is not None:
            wrist_v, tip_v, comp_v = self._match_and_compute_hands_velocity(
                hands_pts, time_scale, timestamp, hand_labels
            )
        elif fallback_velocity is not None:
            comp_v = fallback_velocity * time_scale
            wrist_v = comp_v
            tip_v = comp_v
        else:
            wrist_v, tip_v, comp_v = 0.0, 0.0, 0.0

        # 3. Adaptive Noise Floor
        if comp_v < self.config.velocity_threshold * 0.8:
            self.noise_floor_ema = self.noise_floor_ema * 0.95 + comp_v * 0.05

        effective_threshold = max(self.config.velocity_threshold, self.noise_floor_ema * 2.2)
        is_moving = comp_v >= effective_threshold

        sample = FrameSample(
            timestamp=timestamp,
            frame_index=self.frame_counter,
            feature_vector=list(feature_vector),
            wrist_velocity=wrist_v,
            fingertip_velocity=tip_v,
            composite_velocity=comp_v,
            delta_x=delta_x,
            delta_y=delta_y,
            delta_z=delta_z,
            is_moving=is_moving,
            is_interpolated=False,
            hand_count=hand_count,
        )

        self.sliding_window_buffer.append(sample)
        if len(self.sliding_window_buffer) > self.config.sliding_window_capacity:
            self.sliding_window_buffer.pop(0)

        newly_completed: Optional[SignSegment] = None

        # 4. 3-Phase State Machine (REST <-> MOTION <-> BRIDGING_PAUSE)
        if self.current_phase == "REST":
            if is_moving:
                if self.motion_start_time == 0.0:
                    self.motion_start_time = timestamp
                self.consecutive_motion_frames += 1
                motion_elapsed_ms = max(0.0, timestamp - self.motion_start_time)

                if motion_elapsed_ms >= 50.0:
                    self.current_phase = "MOTION"
                    self.last_rejected_segment = None
                    self.active_segment_start_time = self.motion_start_time or timestamp
                    self.active_segment_start_frame = self.frame_counter

                    onset_anchor = min(self.motion_start_time or timestamp, timestamp)
                    eligible_pre = [
                        s for s in self.sliding_window_buffer
                        if (self.last_boundary_timestamp <= 0 or s.timestamp > self.last_boundary_timestamp)
                        and (self.last_boundary_frame_index < 0 or s.frame_index > self.last_boundary_frame_index)
                        and s.timestamp < timestamp
                        and s.timestamp >= onset_anchor - self.config.pre_motion_padding_ms
                    ]
                    self.active_segment_frames = eligible_pre + [sample]
                    self.consecutive_rest_frames = 0
                    self.bridging_pause_frames = 0
                    self.bridging_pause_start_time = 0.0
            else:
                self.motion_start_time = 0.0
                self.consecutive_motion_frames = 0
                self.consecutive_rest_frames += 1

        elif self.current_phase == "MOTION":
            self.active_segment_frames.append(sample)
            if is_moving:
                self.consecutive_motion_frames += 1
                self.bridging_pause_frames = 0
                self.bridging_pause_start_time = 0.0
                self.consecutive_rest_frames = 0
                if len(self.active_segment_frames) > 1200:
                    self.active_segment_frames.pop(0)
            else:
                self.current_phase = "BRIDGING_PAUSE"
                self.bridging_pause_start_time = timestamp
                self.bridging_pause_frames = 1
                self.consecutive_motion_frames = 0

        elif self.current_phase == "BRIDGING_PAUSE":
            pause_duration_ms = max(0.0, timestamp - self.bridging_pause_start_time)
            if is_moving:
                if pause_duration_ms <= self.config.pause_bridging_grace_ms:
                    self.active_segment_frames.append(sample)
                    self.current_phase = "MOTION"
                    self.bridging_pause_frames = 0
                    self.bridging_pause_start_time = 0.0
                    self.consecutive_motion_frames = 1
                    self.consecutive_rest_frames = 0
                else:
                    trailing_rest = max(0, self.bridging_pause_frames - 2)
                    keep_len = max(1, len(self.active_segment_frames) - trailing_rest)
                    core_samples = self.active_segment_frames[:keep_len]
                    newly_completed = self._evaluate_and_emit_candidate(
                        core_samples, timestamp, effective_threshold
                    )

                    self.last_boundary_timestamp = self.bridging_pause_start_time or (timestamp - 33.3)
                    self.last_boundary_frame_index = self.frame_counter - 1

                    self.current_phase = "MOTION"
                    self.active_segment_start_time = timestamp
                    self.active_segment_start_frame = self.frame_counter
                    self.motion_start_time = timestamp

                    eligible_pre = [
                        s for s in self.sliding_window_buffer
                        if s.timestamp > self.last_boundary_timestamp
                        and s.frame_index > self.last_boundary_frame_index
                        and s.frame_index < self.frame_counter
                        and s.timestamp >= timestamp - self.config.pre_motion_padding_ms
                    ]
                    self.active_segment_frames = eligible_pre + [sample]
                    self.consecutive_motion_frames = 1
                    self.consecutive_rest_frames = 0
                    self.bridging_pause_frames = 0
                    self.bridging_pause_start_time = 0.0
            else:
                self.active_segment_frames.append(sample)
                self.bridging_pause_frames += 1

                required_rest_ms = self.config.pause_bridging_grace_ms + self.config.min_rest_ms
                if pause_duration_ms >= required_rest_ms:
                    self.current_phase = "REST"
                    self.consecutive_motion_frames = 0
                    self.consecutive_rest_frames = self.bridging_pause_frames
                    self.motion_start_time = 0.0

                    trailing_rest = max(0, self.bridging_pause_frames - 2)
                    keep_len = max(1, len(self.active_segment_frames) - trailing_rest)
                    core_samples = self.active_segment_frames[:keep_len]
                    newly_completed = self._evaluate_and_emit_candidate(
                        core_samples, timestamp, effective_threshold
                    )

                    self.last_boundary_timestamp = timestamp
                    self.last_boundary_frame_index = self.frame_counter
                    self.active_segment_frames = []
                    self.bridging_pause_frames = 0
                    self.bridging_pause_start_time = 0.0

        return newly_completed

    def reset(self, preserve_last_rejected: bool = False) -> None:
        self.frame_counter = 0
        self.sliding_window_buffer = []
        self.current_phase = "REST"
        self.consecutive_rest_frames = 0
        self.consecutive_motion_frames = 0
        self.bridging_pause_frames = 0
        self.consecutive_lost_frames = 0
        self.active_segment_frames = []
        self.active_segment_start_time = 0.0
        self.active_segment_start_frame = 0
        self.motion_start_time = 0.0
        self.bridging_pause_start_time = 0.0
        self.lost_start_time = 0.0
        self.prev_hands_by_label = {}
        self.prev_hands_fallback = []
        self.last_timestamp = 0.0
        self.last_boundary_timestamp = -1.0
        self.last_boundary_frame_index = -1
        if not preserve_last_rejected:
            self.last_rejected_segment = None
