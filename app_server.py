"""
Localhost Web Server & Interactive Dual-Mode Studio for VSL Recognition with Full `.npy` Support.

Splits the system into 2 clear parts on http://localhost:8000:
  PART 1: NHẬN DIỆN TỪ ĐƠN (Isolated Word Recognition — BiLSTM)
    - Architecture: `VSLSingleSignBaselineModel` ([B, 30, 138] -> Conv1D(32) -> MaxPool1D(2)
      -> BiLSTM(32x2=64) -> Dropout(0.25) -> Dense(32) -> Softmax)
    - Native `.npy` `(T, 76, 3)` support from `keypoints_splited/{train,test}/<Word>/*.npy`:
      * Extracts 106-D hand geometry/velocity + 32-D Pose NMM = 138-D per frame, resampled to 30 frames.
      * Allows training directly on selected word folders in `keypoints_splited` or testing on any uploaded `.npy` file.

  PART 2: NHẬN DIỆN CÂU LIÊN TỤC (Continuous Sentence Recognition — 2-Layer BiLSTM + CTC)
    - Architecture: `VSLSignCTCModel` ([B, T, 212] -> Conv1D(64) + BatchNorm1D -> 2-Layer
      BiLSTM(128x2=256) -> Linear(64 -> num_classes) -> LogSoftmax + CTCLoss(blank=0))
    - Native `.npy` `(T, 76, 3)` support:
      * Extracts canonical 212-D (`Left 106 + Right 106`) per frame from `seq[:, 33:54, :]` and `seq[:, 54:75, :]`.
      * Composes multi-word continuous sentence training/validation/test sequences strictly within each split (`train/` vs `test/`) with smooth co-articulation transitions.
      * Allows uploading 1 or multiple `.npy` files to concatenate and decode as a continuous sentence via Greedy CTC & Prefix Beam Search CTC.
"""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from decoder import BLANK_ID, BLANK_TOKEN, Vocabulary, normalize_single_gloss
from examples.demo_inference import build_pipeline_verification_dataset
from feature_extractor import (
    CTC_FEATURE_DIM,
    FEATURE_VERSION,
    SINGLE_SIGN_FEATURE_DIM,
    SINGLE_SIGN_SEQUENCE_LENGTH,
    BoundaryDetectionConfig,
    Landmark3D,
    SignBoundaryDetector,
    create_hand_track_state,
    extract_canonical_212_frame,
    extract_invariant_feature_vector,
    normalize_landmarks,
    resample_sequence,
)
from inference import (
    CTCStreamInferenceEngine,
    VSLContinuousRecognizer,
    VSLSingleSignRecognizer,
    export_model_to_torchscript,
)
from train import train_ctc_pipeline, train_single_sign_pipeline
from vsl400_npy_loader import (
    aspect_correct_76x3,
    augment_npy_76x3,
    build_continuous_sentence_samples_from_npy_root,
    find_default_npy_root,
    load_word_samples_from_npy_root,
    npy_sequence_to_212_features,
    npy_sequence_to_word_138_features,
    preprocess_npy_clip_for_ctc,
    scan_npy_dataset_root,
)

DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WEBCAM_NPY_DIR = os.path.join(DEFAULT_DATA_DIR, "webcam_npy")
DEFAULT_WORD_CKPT = os.path.join(DEFAULT_CKPT_DIR, "vsl_word_bilstm.pth")
DEFAULT_SENTENCE_CKPT = os.path.join(DEFAULT_CKPT_DIR, "vsl_sentence_ctc.pth")
DEFAULT_SENTENCE_PT = os.path.join(DEFAULT_CKPT_DIR, "vsl_sentence_ctc.pt")


def _webcam_aspect(body: Dict[str, Any]) -> float:
    """Camera width/height sent by the browser (VSL400 training videos are 224x224 => 1.0)."""
    try:
        a = float(body.get("aspect", 1.0))
    except (TypeError, ValueError):
        a = 1.0
    return a if 0.3 < a < 4.0 else 1.0


def parse_webcam_clip_76x3(body: Dict[str, Any], key: str = "frames_76x3") -> Optional[np.ndarray]:
    raw = body.get(key)
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    clip = np.asarray(raw, dtype=np.float32)
    if clip.ndim != 3 or clip.shape[1:] != (76, 3):
        raise ValueError(f"Mảng webcam clip phải có shape (T, 76, 3), nhận được {clip.shape}")
    clip = np.nan_to_num(clip, nan=0.0, posinf=0.0, neginf=0.0)
    return aspect_correct_76x3(clip, _webcam_aspect(body))


def parse_webcam_frame_76x3(body: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    arr, n = _parse_webcam_frame_76x3_raw(body)
    return aspect_correct_76x3(arr, _webcam_aspect(body)), n


def _parse_webcam_frame_76x3_raw(body: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """
    Convert a webcam JSON payload into a `(76, 3)` float32 array identical to
    `keypoints_extractor.py` (`[pose(33), left_hand(21), right_hand(21), nose(1)]`)
    and return `(frame_76x3, active_hand_count)`.
    """
    raw_76 = body.get("frame_76x3")
    if isinstance(raw_76, list) and len(raw_76) == 76:
        arr = np.asarray(raw_76, dtype=np.float32)
        if arr.shape == (76, 3):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            has_l = int(np.any(np.abs(arr[33:54]) > 1e-6))
            has_r = int(np.any(np.abs(arr[54:75]) > 1e-6))
            return arr, (has_l + has_r)

    # Fallback if legacy `hands` / `handedness` / `pose` payload is sent
    arr = np.zeros((76, 3), dtype=np.float32)
    pose_list = body.get("pose") or []
    for i in range(min(33, len(pose_list))):
        pt = pose_list[i]
        if isinstance(pt, dict):
            arr[i] = [float(pt.get("x", 0.0)), float(pt.get("y", 0.0)), float(pt.get("z", 0.0))]
        elif isinstance(pt, (list, tuple)) and len(pt) >= 3:
            arr[i] = [float(pt[0]), float(pt[1]), float(pt[2])]

    hands = body.get("hands") or []
    handedness = body.get("handedness") or []
    hand_count = 0
    for h_idx, h_pts in enumerate(hands[:2]):
        if not isinstance(h_pts, list) or len(h_pts) < 21:
            continue
        label = str(handedness[h_idx] if h_idx < len(handedness) else "").lower()
        pts_21 = np.zeros((21, 3), dtype=np.float32)
        for j in range(21):
            p = h_pts[j]
            if isinstance(p, dict):
                pts_21[j] = [float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))]
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                pts_21[j] = [float(p[0]), float(p[1]), float(p[2])]
        if "left" in label:
            arr[33:54] = pts_21
        elif "right" in label:
            arr[54:75] = pts_21
        else:
            if not np.any(np.abs(arr[54:75]) > 1e-6):
                arr[54:75] = pts_21
            else:
                arr[33:54] = pts_21
        hand_count += 1

    if np.any(np.abs(arr[0]) > 1e-6):
        arr[75] = arr[0]
    return arr, hand_count


class DualStudioState:
    def __init__(self) -> None:
        os.makedirs(DEFAULT_CKPT_DIR, exist_ok=True)
        os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
        os.makedirs(WEBCAM_NPY_DIR, exist_ok=True)
        self.vocab = Vocabulary()
        self.default_npy_root = (
            find_default_npy_root()
            or r"C:\VSL400\vsl400_video_extracted\keypoints_spliteds"
        )

        # Part 1: Isolated Word Recognition (BiLSTM, 30x138)
        self.word_recognizer: Optional[VSLSingleSignRecognizer] = None
        self.word_is_demo: bool = False
        self.word_live_buffer_76x3: List[np.ndarray] = []
        self.word_missing_hand_streak: int = 0
        self.word_step_counter: int = 0
        self.word_ema_probs: Optional[np.ndarray] = None
        self.last_word_prediction: Optional[Dict[str, Any]] = None
        self.last_captured_clip_76x3: Optional[np.ndarray] = None
        self.word_replay_cache: Optional[List[Any]] = None

        # Part 2: Continuous Sentence Recognition (2-Layer BiLSTM + CTC, Tx212)
        self.sentence_recognizer: Optional[VSLContinuousRecognizer] = None
        self.sentence_stream_engine: Optional[CTCStreamInferenceEngine] = None
        self.sentence_is_demo: bool = False
        self.sentence_live_buffer_76x3: List[np.ndarray] = []
        self.sentence_missing_hand_streak: int = 0
        self.sentence_step_counter: int = 0
        self.last_sentence_prediction: Optional[Dict[str, Any]] = None

        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        if os.path.exists(DEFAULT_WORD_CKPT):
            try:
                self.word_recognizer = VSLSingleSignRecognizer(DEFAULT_WORD_CKPT)
            except Exception as e:
                print(f"[SERVER] Could not load word checkpoint: {e}")

        if os.path.exists(DEFAULT_SENTENCE_CKPT):
            try:
                self.sentence_recognizer = VSLContinuousRecognizer(DEFAULT_SENTENCE_CKPT)
                self.sentence_stream_engine = CTCStreamInferenceEngine(
                    self.sentence_recognizer,
                    window_size_frames=75,
                    stride_frames=6,
                    min_frames_to_infer=12,
                )
                self.vocab = self.sentence_recognizer.vocab
            except Exception as e:
                print(f"[SERVER] Could not load sentence checkpoint: {e}")

    def reload_word_checkpoint(self, path: str, is_demo: bool = False) -> None:
        self.word_recognizer = VSLSingleSignRecognizer(path)
        self.word_is_demo = is_demo
        self.word_ema_probs = None
        self.word_replay_cache = None

    def reload_sentence_checkpoint(self, path: str, is_demo: bool = False) -> None:
        self.sentence_recognizer = VSLContinuousRecognizer(path)
        self.vocab = self.sentence_recognizer.vocab
        self.sentence_stream_engine = CTCStreamInferenceEngine(
            self.sentence_recognizer,
            window_size_frames=75,
            stride_frames=6,
            min_frames_to_infer=12,
        )
        self.sentence_is_demo = is_demo

    def ensure_word_model_loaded(self) -> None:
        if self.word_recognizer is None:
            if os.path.isdir(self.default_npy_root):
                word_samples = load_word_samples_from_npy_root(
                    root_dir=self.default_npy_root,
                    max_classes=8,
                    max_files_per_class_split=35,
                )
                train_single_sign_pipeline(
                    rows=word_samples,
                    output_model_path=DEFAULT_WORD_CKPT,
                    epochs=35,
                    batch_size=8,
                    lr=1.5e-3,
                    seed=42,
                )
                self.reload_word_checkpoint(DEFAULT_WORD_CKPT, is_demo=False)
            else:
                _, single_samples = build_pipeline_verification_dataset()
                train_single_sign_pipeline(
                    rows=single_samples,
                    output_model_path=DEFAULT_WORD_CKPT,
                    epochs=4,
                    batch_size=4,
                    seed=42,
                )
                self.reload_word_checkpoint(DEFAULT_WORD_CKPT, is_demo=True)

    def ensure_sentence_model_loaded(self) -> None:
        if self.sentence_recognizer is None:
            if os.path.isdir(self.default_npy_root):
                samples, custom_vocab = build_continuous_sentence_samples_from_npy_root(
                    root_dir=self.default_npy_root,
                    max_classes=6,
                    sentences_per_split=100,
                )
                train_ctc_pipeline(
                    raw_samples=samples,
                    output_model_path=DEFAULT_SENTENCE_CKPT,
                    vocab=custom_vocab,
                    epochs=30,
                    batch_size=8,
                    lr=1.5e-3,
                    seed=42,
                )
                self.reload_sentence_checkpoint(DEFAULT_SENTENCE_CKPT, is_demo=False)
            else:
                ctc_samples, _ = build_pipeline_verification_dataset()
                train_ctc_pipeline(
                    raw_samples=ctc_samples,
                    output_model_path=DEFAULT_SENTENCE_CKPT,
                    vocab=self.vocab,
                    epochs=3,
                    batch_size=4,
                    seed=42,
                )
                self.reload_sentence_checkpoint(DEFAULT_SENTENCE_CKPT, is_demo=True)

    def reset_word_stream(self) -> None:
        self.word_live_buffer_76x3 = []
        self.word_missing_hand_streak = 0
        self.word_step_counter = 0
        self.word_ema_probs = None
        self.last_word_prediction = None

    def reset_sentence_stream(self) -> None:
        if self.sentence_stream_engine is not None:
            self.sentence_stream_engine.reset()
        self.sentence_live_buffer_76x3 = []
        self.sentence_missing_hand_streak = 0
        self.sentence_step_counter = 0
        self.last_sentence_prediction = None


STATE = DualStudioState()


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VSL Studio (.npy keypoints_splited &amp; Holistic Webcam) — Phần 1: Từ (BiLSTM) &amp; Phần 2: Câu (CTC)</title>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js" crossorigin="anonymous"></script>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --purple: #a855f7;
      --success: #22c55e;
      --warn: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    header {
      padding: 16px 28px;
      background: #020617;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    header h1 { margin: 0; font-size: 1.2rem; color: var(--accent); }
    .tabs {
      max-width: 1360px;
      margin: 16px auto 0;
      padding: 0 20px;
      display: flex;
      gap: 12px;
    }
    .tab-btn {
      flex: 1;
      padding: 14px 20px;
      font-size: 0.98rem;
      font-weight: 700;
      border-radius: 10px;
      border: 2px solid var(--border);
      background: #1e293b;
      color: var(--muted);
      cursor: pointer;
      text-align: left;
      transition: all 0.15s;
    }
    .tab-btn.active-word {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.12);
      color: var(--text);
    }
    .tab-btn.active-sentence {
      border-color: var(--purple);
      background: rgba(168, 85, 247, 0.14);
      color: var(--text);
    }
    .tab-sub {
      display: block;
      font-size: 0.8rem;
      font-weight: 400;
      color: var(--muted);
      margin-top: 3px;
    }
    .container {
      max-width: 1360px;
      margin: 0 auto;
      padding: 16px 20px 28px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 980px) {
      .container { grid-template-columns: 1fr; }
    }
    .hidden { display: none !important; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
    }
    .card h2 {
      margin-top: 0;
      font-size: 1.03rem;
      color: var(--accent);
      border-bottom: 1px solid var(--border);
      padding-bottom: 9px;
    }
    .card.sentence-card h2 { color: #c084fc; }
    button {
      background: var(--accent);
      color: #020617;
      border: none;
      padding: 8px 14px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }
    button.purple { background: #a855f7; color: white; }
    button.green { background: #22c55e; color: #020617; }
    button.secondary { background: #334155; color: var(--text); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    pre {
      background: #020617;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      overflow-x: auto;
      font-size: 0.8rem;
      max-height: 270px;
      overflow-y: auto;
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    input[type="number"], input[type="text"], select, input[type="file"] {
      background: #020617;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 6px 10px;
      border-radius: 6px;
    }
    .arch-box {
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 0.82rem;
      margin-bottom: 12px;
      color: #cbd5e1;
    }
    .result-box {
      padding: 12px 14px;
      background: #020617;
      border-radius: 8px;
      border: 1px solid var(--border);
      margin-bottom: 10px;
    }
    .word-badge {
      display: inline-block;
      padding: 6px 14px;
      border-radius: 8px;
      background: #0284c7;
      color: white;
      font-weight: 700;
      font-size: 1.12rem;
    }
    .gloss-chip {
      display: inline-block;
      padding: 4px 10px;
      margin: 3px;
      background: #9333ea;
      color: white;
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.92rem;
    }
    .video-wrap {
      position: relative;
      width: 420px;
      max-width: 100%;
      aspect-ratio: 4/3;
      background: #020617;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    video { display: none; }
    canvas { width: 100%; height: 100%; object-fit: cover; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>VSL Recognition Studio — Hỗ trợ trực tiếp dữ liệu <code>.npy</code> (<code>keypoints_splited</code>)</h1>
      <div style="font-size: 0.81rem; color: var(--muted);">
        Định dạng file <code>.npy</code>: <code>(T, 76, 3)</code> = 33 Pose + 21 Tay trái + 21 Tay phải + 1 Mũi &rarr; Tự động trích xuất cho <strong>Phần 1 (Từ - BiLSTM)</strong> &amp; <strong>Phần 2 (Câu - CTC)</strong>
      </div>
    </div>
    <div id="headerStats" style="font-size:0.84rem; color:var(--muted);">Đang tải...</div>
  </header>

  <!-- CONFIGURATION BAR FOR KEYPOINTS_SPLITED ROOT -->
  <div style="max-width:1360px; margin: 14px auto 0; padding: 0 20px;">
    <div class="card" style="padding: 12px 18px;">
      <div class="row" style="margin-bottom:0;">
        <strong>Thư mục <code>keypoints_splited</code> (.npy):</strong>
        <input type="text" id="npyRootInput" style="flex:1; min-width:340px;" value="C:\VSL400\SignCTCModel\archive\processed\processed\keypoints_splited" />
        <button onclick="scanNpyFolder()" class="secondary">Quét danh sách Từ (.npy)</button>
        <span id="npyScanSummary" style="font-size:0.84rem; color:var(--accent);"></span>
      </div>
    </div>
  </div>

  <!-- TWO MAIN TABS -->
  <div class="tabs">
    <button id="tabBtnWord" class="tab-btn active-word" onclick="switchMode('word')">
      PHẦN 1: NHẬN DIỆN TỪ ĐƠN (ISOLATED WORD — BiLSTM)
      <span class="tab-sub">Đọc trực tiếp .npy (T, 76, 3) &rarr; [30 &times; 138] &rarr; Conv1D(32) &rarr; MaxPool1D(2) &rarr; BiLSTM(32&times;2=64) &rarr; Dense(32) &rarr; Softmax</span>
    </button>
    <button id="tabBtnSentence" class="tab-btn" onclick="switchMode('sentence')">
      PHẦN 2: NHẬN DIỆN CÂU LIÊN TỤC (CONTINUOUS SENTENCE — 2-Layer BiLSTM + CTC)
      <span class="tab-sub">Ghép các từ .npy (T, 76, 3) thành câu liên tục [T_total &times; 212] &rarr; Conv1D(64) &rarr; 2-Layer BiLSTM(128&times;2=256) &rarr; CTCLoss(blank=0)</span>
    </button>
  </div>

  <!-- ===================================================================== -->
  <!-- PART 1: ISOLATED WORD RECOGNITION (BiLSTM)                            -->
  <!-- ===================================================================== -->
  <div id="panelWord" class="container">
    <div class="card">
      <h2>1A. Huấn luyện Model Nhận diện Từ đơn (BiLSTM) từ <code>keypoints_splited</code> (.npy)</h2>
      <div class="arch-box">
        <strong>Pipeline chuyển đổi <code>.npy (T, 76, 3)</code> sang BiLSTM Từ đơn:</strong><br/>
        • Tách <code>left_hand = seq[:, 33:54]</code>, <code>right_hand = seq[:, 54:75]</code>, <code>pose = seq[:, 0:33]</code>.<br/>
        • Trích xuất <code>138 đặc trưng/frame</code> (106 hình học &amp; vận tốc bàn tay + 32 NMM cơ thể), nội suy về <code>(30, 138)</code>.<br/>
        • Đưa qua mạng <strong>BiLSTM</strong> (<code>VSLSingleSignBaselineModel</code>) để phân loại từ.
      </div>
      <div class="row">
        <label>Các từ muốn huấn luyện (cách nhau dấu phẩy, để trống = lấy tự động):</label>
        <input type="text" id="wordSelectedFolders" style="width:100%;" placeholder="Ví dụ: An ủi, Anh, Ba lô, Ban ngày, Ban đêm (Để trống sẽ lấy N từ đầu tiên)" />
      </div>
      <div class="row">
        <label>Số lớp Từ (nếu để trống ô trên):</label>
        <input type="number" id="wordMaxClasses" value="8" min="2" max="472" style="width:70px;" />
        <label>Số file .npy tối đa / từ:</label>
        <input type="number" id="wordMaxFiles" value="35" min="5" max="200" style="width:70px;" />
      </div>
      <div class="row">
        <label>Epochs:</label>
        <input type="number" id="wordEpochs" value="35" min="1" max="300" style="width:70px;" />
        <label>Batch Size:</label>
        <input type="number" id="wordBatch" value="8" min="1" max="128" style="width:70px;" />
        <label>LR:</label>
        <input type="number" id="wordLr" value="0.0015" step="0.0005" style="width:85px;" />
      </div>
      <div class="row">
        <button id="btnTrainWordNpy" class="green" onclick="trainWordFromNpy()">Huấn luyện BiLSTM Từ đơn từ thư mục .npy</button>
      </div>
      <div id="wordTrainMsg" style="font-size:0.85rem; color:var(--accent); margin-bottom:6px;"></div>
      <pre id="wordTrainOutput">Bấm "Huấn luyện BiLSTM Từ đơn từ thư mục .npy" để huấn luyện trên dữ liệu thực tế trong keypoints_splited.</pre>
    </div>

    <div class="card">
      <h2>1B. Nhận diện Từ đơn (BiLSTM) từ file <code>.npy</code></h2>
      <div class="arch-box">
        Bạn có thể <strong>chọn trực tiếp 1 file <code>.npy</code> bất kỳ</strong> trong <code>keypoints_splited/test/&lt;Tên_Từ&gt;/*.npy</code> (thuộc các lớp từ đã huấn luyện) để kiểm tra mạng BiLSTM dự đoán ra từ gì.
      </div>
      <div class="row">
        <label>Chọn 1 file <code>.npy</code> từ máy:</label>
        <input type="file" id="wordNpyFileInput" accept=".npy" />
        <button onclick="predictWordFromUploadedNpy()">Nhận diện file .npy này (BiLSTM)</button>
      </div>
      <div class="row">
        <button class="secondary" onclick="predictRandomTestNpyWord()">Lấy ngẫu nhiên 1 file .npy trong tập Test để nhận diện</button>
      </div>
      <div class="result-box">
        <div style="font-size:0.82rem; color:var(--muted);">Kết quả dự đoán của mạng BiLSTM:</div>
        <div id="wordSampleBadge" style="margin-top:8px;">—</div>
      </div>
      <pre id="wordPredictOutput">Chọn file .npy hoặc bấm lấy ngẫu nhiên 1 file .npy trong tập Test để xem chi tiết xác suất.</pre>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- PART 2: CONTINUOUS SENTENCE RECOGNITION (BiLSTM + CTC)                -->
  <!-- ===================================================================== -->
  <div id="panelSentence" class="container hidden">
    <div class="card sentence-card">
      <h2>2A. Huấn luyện Nhận diện Câu liên tục (2-Layer BiLSTM + CTC) từ <code>keypoints_splited</code> (.npy)</h2>
      <div class="arch-box">
        <strong>Vì sao ghép được Câu liên tục từ các file <code>.npy</code> từ đơn?</strong><br/>
        • Hệ thống đọc các file <code>.npy (T, 76, 3)</code>, nội suy các frame bị khuyết tay, cắt bỏ đầu/cuối rỗng và trích xuất <code>212 đặc trưng chuẩn/frame</code> (<code>Tay trái 106 + Tay phải 106</code>).<br/>
        • Ghép 2–3 từ liên tiếp (kèm nội suy chuyển tiếp mượt co-articulation giữa các từ) thành câu liên tục <code>[T_total, 212]</code>.<br/>
        • <strong>Đảm bảo không rò rỉ dữ liệu:</strong> Câu huấn luyện chỉ ghép từ thư mục <code>train/</code>; câu kiểm thử chỉ ghép từ thư mục <code>test/</code>.<br/>
        • Huấn luyện bằng <code>VSLSignCTCModel</code> (<code>Conv1D(64) &rarr; 2-Layer BiLSTM(256) &rarr; CTCLoss(blank=0)</code>).
      </div>
      <div class="row">
        <label>Các từ dùng để ghép câu (để trống = lấy tự động các từ giàu dữ liệu):</label>
        <input type="text" id="sentSelectedFolders" style="width:100%;" placeholder="Ví dụ: An ủi, Anh, Ba lô, Bia, Biết, Biếu tặng" />
      </div>
      <div class="row">
        <label>Số lớp Từ:</label>
        <input type="number" id="sentMaxClasses" value="6" min="2" max="100" style="width:65px;" />
        <label>Số câu sinh ra:</label>
        <input type="number" id="sentCount" value="100" min="15" max="500" style="width:70px;" />
        <label>Epochs:</label>
        <input type="number" id="sentenceEpochs" value="30" min="1" max="300" style="width:65px;" />
        <label>Batch:</label>
        <input type="number" id="sentenceBatch" value="8" min="1" max="128" style="width:60px;" />
      </div>
      <div class="row">
        <button id="btnTrainSentNpy" class="purple" onclick="trainSentenceFromNpy()">Huấn luyện 2-Layer BiLSTM + CTC từ thư mục .npy</button>
      </div>
      <div id="sentenceTrainMsg" style="font-size:0.85rem; color:#c084fc; margin-bottom:6px;"></div>
      <pre id="sentenceTrainOutput">Bấm nút trên để ghép câu từ keypoints_splited (.npy) và huấn luyện model CTC.</pre>
    </div>

    <div class="card sentence-card">
      <h2>2B. Giải mã Câu liên tục (Greedy &amp; Beam Search CTC) từ các file <code>.npy</code></h2>
      <div class="arch-box">
        Bạn có thể <strong>chọn 1 hoặc nhiều file <code>.npy</code> cùng lúc</strong> (ví dụ chọn 2–3 file <code>.npy</code> của 2–3 từ khác nhau): hệ thống sẽ nối chúng thành một chuỗi câu liên tục không phân đoạn và dùng <strong>CTC Decoder</strong> để giải mã ra chuỗi từ!
      </div>
      <div class="row">
        <label>Chọn 1 hoặc nhiều file <code>.npy</code>:</label>
        <input type="file" id="sentNpyFilesInput" accept=".npy" multiple />
      </div>
      <div class="row">
        <button class="purple" onclick="predictSentenceFromUploadedNpy('greedy')">Giải mã file .npy đã chọn (Greedy CTC)</button>
        <button class="purple" onclick="predictSentenceFromUploadedNpy('beam_search')">Giải mã file .npy đã chọn (Beam Search CTC)</button>
        <button class="secondary" onclick="predictRandomTestSentenceNpy()">Ghép ngẫu nhiên 1 câu từ tập Test (.npy) &amp; Giải mã</button>
      </div>
      <div class="result-box">
        <div style="font-size:0.82rem; color:var(--muted);">Chuỗi từ giải mã được (CTC Decoded Gloss Sequence):</div>
        <div id="sentDecodedChips" style="margin-top:8px;">—</div>
      </div>
      <pre id="sentencePredictOutput">Chọn các file .npy hoặc bấm "Ghép ngẫu nhiên 1 câu từ tập Test (.npy)" để kiểm tra giải mã CTC.</pre>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- UNIFIED HOLISTIC (76, 3) WEBCAM REALTIME & RECORDING PANEL            -->
  <!-- ===================================================================== -->
  <div class="container" style="padding-top: 0;">
    <div class="card" style="grid-column: 1 / -1;">
      <h2 id="webcamTitle">3. Nhận diện Thực tế qua Webcam (MediaPipe Holistic 76 điểm chuẩn <code>(T, 76, 3)</code>)</h2>
      <div class="arch-box">
        <strong>Đồng bộ 100% công thức giữa file <code>.npy</code> và Webcam thật:</strong><br/>
        • Webcam dùng <strong>MediaPipe Holistic</strong> trích xuất đúng <strong>76 điểm <code>(76, 3)</code>/frame</strong>: <code>33 điểm Pose + 21 điểm Tay Trái (Màu Xanh Lá) + 21 điểm Tay Phải (Màu Xanh Dương) + 1 điểm Mũi</code> — giống hệt <code>keypoints_extractor.py</code>!<br/>
        • <strong>Mẹo quan trọng:</strong> Hãy <strong>ĐỂ TẮT</strong> ô "Đảo gương". Dữ liệu train <code>.npy</code> được trích từ video gốc không lật gương, đúng như khung hình thô của webcam (hình hiển thị không lật gương nên trông như bị ngược — đó là đúng). Holistic tự xác định tay theo cơ thể: <strong>tay phải thật của bạn phải hiện nhãn R (Xanh Dương)</strong>. Đứng cách camera ~1–1.5m sao cho thấy cả <strong>vai và đầu</strong>, giống video mẫu.
      </div>
      <div class="row">
        <button id="btnStartCam" onclick="startWebcam()">1. Bật Webcam (Holistic 76 điểm)</button>
        <button id="btnStopCam" class="secondary" onclick="stopWebcam()" disabled>Tắt Webcam</button>
        <button id="btnRecordClip" class="green" onclick="captureCleanClipAndPredict()" disabled>🎥 2. Quay chuẩn 1 Ký hiệu (2.5 giây) &amp; Nhận diện</button>
        <button class="secondary" onclick="resetActiveStream()">Xoá Bộ đệm</button>
        <label style="display:flex; align-items:center; gap:6px; color:#f8fafc; cursor:pointer; background:#020617; padding:6px 12px; border-radius:8px; border:1px solid var(--border);">
          <input type="checkbox" id="chkMirrorFlip" />
          Đảo gương Trái/Phải (Mirror Flip X &amp; Swap L/R)
        </label>
        <span id="camStatusText" style="font-size:0.86rem; font-weight:600; color:var(--accent);"></span>
      </div>
      <div style="display:grid; grid-template-columns: auto 1fr; gap: 20px; align-items: start;">
        <div class="video-wrap">
          <video id="webcamVideo" playsinline></video>
          <canvas id="webcamCanvas" width="640" height="480"></canvas>
        </div>
        <div>
          <div class="result-box">
            <div style="font-size:0.82rem; color:var(--muted);">Trạng thái Luồng <code>(T, 76, 3)</code> trên Webcam:</div>
            <div id="livePhaseText" style="font-size:1.02rem; font-weight:700; color:var(--accent);">Chưa bật Webcam</div>
          </div>
          <div id="liveOutputBox" class="result-box">
            <div style="font-size:0.82rem; color:var(--muted);">Kết quả nhận diện trực tiếp từ Webcam:</div>
            <div id="liveResultContent" style="margin-top:8px;">—</div>
          </div>
          <div class="result-box" style="border-color:#334155;">
            <div style="font-size:0.82rem; color:#38bdf8; font-weight:600; margin-bottom:6px;">
              ➕ Tinh chỉnh nhanh theo góc Camera của bạn (Personal Webcam Adaptation):
            </div>
            <div style="font-size:0.78rem; color:var(--muted); margin-bottom:8px;">
              Nếu góc ngồi / khoảng cách camera của bạn khác với video mẫu, hãy thực hiện ký hiệu trên Webcam rồi chọn nhãn đúng và bấm nút dưới để model học ngay góc máy của bạn trong 3 giây:
            </div>
            <div class="row" style="margin-bottom:0;">
              <label>Nhãn của ký hiệu vừa thực hiện:</label>
              <select id="webcamAdaptLabelSelect" style="min-width:140px;"></select>
              <button id="btnFinetuneCam" class="purple" onclick="finetuneWithLastWebcamClip()">Lưu mẫu Webcam vừa quay &amp; Tinh chỉnh nhanh Model (3s)</button>
            </div>
            <div id="webcamAdaptStatus" style="font-size:0.82rem; color:#22c55e; margin-top:6px;"></div>
          </div>
          <pre id="liveTelemetry" style="max-height:140px;">Telemetry trực tiếp.</pre>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentMode = 'word';

    function switchMode(mode) {
      currentMode = mode;
      document.getElementById('panelWord').classList.toggle('hidden', mode !== 'word');
      document.getElementById('panelSentence').classList.toggle('hidden', mode !== 'sentence');
      document.getElementById('tabBtnWord').className = mode === 'word' ? 'tab-btn active-word' : 'tab-btn';
      document.getElementById('tabBtnSentence').className = mode === 'sentence' ? 'tab-btn active-sentence' : 'tab-btn';
    }

    function arrayBufferToBase64(buffer) {
      let binary = '';
      const bytes = new Uint8Array(buffer);
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
      }
      return btoa(binary);
    }

    async function refreshStatus() {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.default_npy_root) {
        document.getElementById('npyRootInput').value = data.default_npy_root;
      }
      document.getElementById('headerStats').innerHTML =
        `Model Từ (BiLSTM): <strong>${data.word_model_loaded ? ('Đã nạp (' + data.word_classes.length + ' từ: ' + data.word_classes.join(', ') + ')') : 'Chưa nạp'}</strong> | ` +
        `Model Câu (CTC): <strong>${data.sentence_model_loaded ? ('Đã nạp (' + (data.sentence_num_classes - 1) + ' từ)') : 'Chưa nạp'}</strong>`;

      const sel = document.getElementById('webcamAdaptLabelSelect');
      if (sel && data.word_classes && data.word_classes.length > 0) {
        const curVal = sel.value;
        sel.innerHTML = data.word_classes.map(c => `<option value="${c}">${c}</option>`).join('');
        if (curVal && data.word_classes.includes(curVal)) sel.value = curVal;
      }
    }

    async function scanNpyFolder() {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      const res = await fetch('/api/npy/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ root_dir })
      });
      const data = await res.json();
      if (data.ok) {
        document.getElementById('npyScanSummary').textContent =
          `Tìm thấy ${data.info.total_classes} lớp từ! Các lớp giàu dữ liệu mặc định: ${(data.info.recommended_folders || []).slice(0, 8).join(', ')}`;
      } else {
        document.getElementById('npyScanSummary').textContent = `Lỗi: ${data.error}`;
      }
    }

    function parseSelectedFolders(inputId) {
      const raw = document.getElementById(inputId).value.trim();
      if (!raw) return null;
      return raw.split(',').map(s => s.trim()).filter(Boolean);
    }

    // ---------------- PART 1: WORD (BiLSTM) ----------------
    async function trainWordFromNpy() {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      const selected_folders = parseSelectedFolders('wordSelectedFolders');
      const max_classes = parseInt(document.getElementById('wordMaxClasses').value || '8', 10);
      const max_files = parseInt(document.getElementById('wordMaxFiles').value || '35', 10);
      const epochs = parseInt(document.getElementById('wordEpochs').value || '35', 10);
      const batch_size = parseInt(document.getElementById('wordBatch').value || '8', 10);
      const lr = parseFloat(document.getElementById('wordLr').value || '0.0015');

      const btn = document.getElementById('btnTrainWordNpy');
      btn.disabled = true;
      document.getElementById('wordTrainMsg').textContent =
        'Đang đọc các file .npy từ keypoints_splited, nội suy khuyết tay, cân bằng lớp (30x138) và huấn luyện BiLSTM...';

      try {
        const res = await fetch('/api/word/train_npy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            root_dir,
            selected_folders,
            max_classes,
            max_files_per_class: max_files,
            epochs,
            batch_size,
            lr
          })
        });
        const data = await res.json();
        document.getElementById('wordTrainOutput').textContent = JSON.stringify(data, null, 2);
        if (data.ok) {
          const valAcc = (data.summary.best_val_accuracy * 100).toFixed(1);
          const testAcc = (data.summary.test_metrics.accuracy * 100).toFixed(1);
          const macroF1 = (data.summary.test_metrics.macro_f1 * 100).toFixed(1);
          document.getElementById('wordTrainMsg').textContent =
            `Huấn luyện BiLSTM hoàn tất (${data.summary.class_names.join(', ')})! Val Acc = ${valAcc}% | Test Acc = ${testAcc}% | Macro-F1 = ${macroF1}%`;
          await refreshStatus();
        } else {
          document.getElementById('wordTrainMsg').textContent = `Lỗi: ${data.error}`;
        }
      } finally {
        btn.disabled = false;
      }
    }

    function renderWordBadge(res, groundTruth) {
      if (!res) return '—';
      const color = res.status === 'ACCEPTED' ? '#16a34a' : (res.status === 'AMBIGUOUS' ? '#d97706' : '#dc2626');
      const top3 = (res.top3 || []).map(t => `${t.label}: ${(t.prob * 100).toFixed(1)}%`).join(' | ');
      const gtHtml = groundTruth ? `<div style="margin-bottom:6px; font-size:0.86rem; color:#38bdf8;">Nguồn / Nhãn: <strong>${groundTruth}</strong></div>` : '';
      return `
        ${gtHtml}
        <span class="word-badge" style="background:${color};">${res.predicted_label}</span>
        <span style="margin-left:10px; font-size:0.9rem;">
          Top-1 gốc: <strong>${res.raw_top1_label}</strong> |
          Độ tự tin: <strong>${(res.confidence * 100).toFixed(1)}%</strong> |
          Trạng thái: <code>${res.status}</code>
        </span>
        <div style="margin-top:6px; font-size:0.82rem; color:var(--muted);">Top-3 xác suất: ${top3}</div>
      `;
    }

    async function predictWordFromUploadedNpy() {
      const fInput = document.getElementById('wordNpyFileInput');
      if (!fInput.files || fInput.files.length === 0) {
        alert('Hãy chọn 1 file .npy trên máy tính của bạn.');
        return;
      }
      const file = fInput.files[0];
      const b64 = arrayBufferToBase64(await file.arrayBuffer());
      const res = await fetch('/api/word/predict_npy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name, npy_base64: b64 })
      });
      const data = await res.json();
      document.getElementById('wordPredictOutput').textContent = JSON.stringify(data, null, 2);
      if (data.ok) {
        document.getElementById('wordSampleBadge').innerHTML = renderWordBadge(data.result, data.filename);
      }
    }

    async function predictRandomTestNpyWord() {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      const res = await fetch('/api/word/predict_random_test_npy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ root_dir })
      });
      const data = await res.json();
      document.getElementById('wordPredictOutput').textContent = JSON.stringify(data, null, 2);
      if (data.ok) {
        document.getElementById('wordSampleBadge').innerHTML =
          renderWordBadge(data.result, `${data.true_label} (${data.file_path})`);
      }
    }

    // ---------------- PART 2: SENTENCE (2-LAYER BiLSTM + CTC) ----------------
    async function trainSentenceFromNpy() {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      const selected_folders = parseSelectedFolders('sentSelectedFolders');
      const max_classes = parseInt(document.getElementById('sentMaxClasses').value || '6', 10);
      const sentences_per_split = parseInt(document.getElementById('sentCount').value || '100', 10);
      const epochs = parseInt(document.getElementById('sentenceEpochs').value || '30', 10);
      const batch_size = parseInt(document.getElementById('sentenceBatch').value || '8', 10);

      const btn = document.getElementById('btnTrainSentNpy');
      btn.disabled = true;
      document.getElementById('sentenceTrainMsg').textContent =
        'Đang ghép các câu liên tục từ keypoints_splited (.npy), trích xuất [T, 212] và huấn luyện 2-Layer BiLSTM + CTC...';

      try {
        const res = await fetch('/api/sentence/train_npy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            root_dir,
            selected_folders,
            max_classes,
            sentences_per_split,
            epochs,
            batch_size
          })
        });
        const data = await res.json();
        document.getElementById('sentenceTrainOutput').textContent = JSON.stringify(data, null, 2);
        if (data.ok) {
          const valWer = (data.summary.best_val_wer * 100).toFixed(1);
          const testWer = (data.summary.test_metrics.corpus_wer * 100).toFixed(1);
          document.getElementById('sentenceTrainMsg').textContent =
            `Huấn luyện CTC hoàn tất! Val WER = ${valWer}% | Test WER = ${testWer}% | Exact Match = ${(data.summary.test_metrics.exact_match_rate * 100).toFixed(1)}%`;
          await refreshStatus();
        } else {
          document.getElementById('sentenceTrainMsg').textContent = `Lỗi: ${data.error}`;
        }
      } finally {
        btn.disabled = false;
      }
    }

    function renderSentenceChips(decodedGlosses, trueGlosses) {
      const gtHtml = trueGlosses
        ? `<div style="margin-bottom:6px; font-size:0.86rem; color:#38bdf8;">Câu gốc (Ground Truth): <strong>${trueGlosses.join(' &rarr; ')}</strong></div>`
        : '';
      const chips = (decodedGlosses && decodedGlosses.length > 0)
        ? decodedGlosses.map(g => `<span class="gloss-chip">${g}</span>`).join(' &rarr; ')
        : '<em>&lt;blank&gt; (Chưa phát hiện từ)</em>';
      return `${gtHtml}<div>Dự đoán CTC: ${chips}</div>`;
    }

    async function predictSentenceFromUploadedNpy(decodeMethod) {
      const fInput = document.getElementById('sentNpyFilesInput');
      if (!fInput.files || fInput.files.length === 0) {
        alert('Hãy chọn 1 hoặc nhiều file .npy trên máy tính để ghép thành câu và giải mã CTC.');
        return;
      }
      const filesPayload = [];
      for (const f of fInput.files) {
        filesPayload.push({
          filename: f.name,
          npy_base64: arrayBufferToBase64(await f.arrayBuffer())
        });
      }
      const res = await fetch('/api/sentence/predict_npy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decode_method: decodeMethod, files: filesPayload })
      });
      const data = await res.json();
      document.getElementById('sentencePredictOutput').textContent = JSON.stringify(data, null, 2);
      if (data.ok) {
        document.getElementById('sentDecodedChips').innerHTML =
          renderSentenceChips(data.result.predicted_glosses, null);
      }
    }

    async function predictRandomTestSentenceNpy() {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      const res = await fetch('/api/sentence/predict_random_test_npy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ root_dir })
      });
      const data = await res.json();
      document.getElementById('sentencePredictOutput').textContent = JSON.stringify(data, null, 2);
      if (data.ok) {
        document.getElementById('sentDecodedChips').innerHTML =
          renderSentenceChips(data.greedy_result.predicted_glosses, data.true_glosses);
      }
    }

    // ---------------- WEBCAM (HOLISTIC 76x3 UNIFIED WITH KEYPOINTS_EXTRACTOR.PY) ----------------
    let cameraObj = null;
    let holisticObj = null;
    let isSendingFrame = false;
    let isRecordingClip = false;
    let recordedClip76x3 = [];
    let lastCapturedClip76x3 = [];

    // Pose pairs to swap when mirrorFlip is enabled: (1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)
    const POSE_LR_PAIRS = [
      [1,4],[2,5],[3,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,22],[23,24],[25,26],[27,28],[29,30],[31,32]
    ];

    // Real camera aspect (width/height). VSL400 training videos are 224x224, so the server
    // rescales webcam x-coordinates by this factor to undo the 4:3 / 16:9 anisotropy.
    function camAspect() {
      const v = document.getElementById('webcamVideo');
      return (v && v.videoWidth > 0 && v.videoHeight > 0) ? (v.videoWidth / v.videoHeight) : (4 / 3);
    }

    function extract76x3FromHolisticResults(results, mirrorFlip) {
      const pose33 = Array.from({ length: 33 }, () => [0.0, 0.0, 0.0]);
      const left21 = Array.from({ length: 21 }, () => [0.0, 0.0, 0.0]);
      const right21 = Array.from({ length: 21 }, () => [0.0, 0.0, 0.0]);
      let nose1 = [[0.0, 0.0, 0.0]];

      if (results.poseLandmarks && results.poseLandmarks.length >= 33) {
        for (let i = 0; i < 33; i++) {
          const lm = results.poseLandmarks[i];
          pose33[i] = [
            mirrorFlip ? (1.0 - lm.x) : lm.x,
            lm.y,
            lm.z || 0.0
          ];
        }
        if (mirrorFlip) {
          for (const [a, b] of POSE_LR_PAIRS) {
            const tmp = pose33[a];
            pose33[a] = pose33[b];
            pose33[b] = tmp;
          }
        }
        nose1 = [[pose33[0][0], pose33[0][1], pose33[0][2]]];
      }
      // keypoints_extractor.py uses face_landmarks[1] (nose tip of FaceMesh) for the 76th point
      if (results.faceLandmarks && results.faceLandmarks.length > 1) {
        const f = results.faceLandmarks[1];
        nose1 = [[mirrorFlip ? (1.0 - f.x) : f.x, f.y, f.z || 0.0]];
      }

      const rawLeft = mirrorFlip ? results.rightHandLandmarks : results.leftHandLandmarks;
      const rawRight = mirrorFlip ? results.leftHandLandmarks : results.rightHandLandmarks;

      if (rawLeft && rawLeft.length >= 21) {
        for (let i = 0; i < 21; i++) {
          const lm = rawLeft[i];
          left21[i] = [mirrorFlip ? (1.0 - lm.x) : lm.x, lm.y, lm.z || 0.0];
        }
      }
      if (rawRight && rawRight.length >= 21) {
        for (let i = 0; i < 21; i++) {
          const lm = rawRight[i];
          right21[i] = [mirrorFlip ? (1.0 - lm.x) : lm.x, lm.y, lm.z || 0.0];
        }
      }

      const hasLeft = rawLeft && rawLeft.length >= 21;
      const hasRight = rawRight && rawRight.length >= 21;
      const frame76 = [...pose33, ...left21, ...right21, ...nose1];
      return { frame76, hasLeft: !!hasLeft, hasRight: !!hasRight, rawLeft, rawRight };
    }

    async function resetActiveStream() {
      await fetch('/api/stream_reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: currentMode })
      });
      document.getElementById('liveResultContent').textContent = '—';
    }

    async function captureCleanClipAndPredict() {
      if (!cameraObj || isRecordingClip) return;
      const btn = document.getElementById('btnRecordClip');
      btn.disabled = true;
      recordedClip76x3 = [];

      const durationSec = currentMode === 'word' ? 2.4 : 4.0;
      document.getElementById('camStatusText').textContent = 'Chuẩn bị giơ tay... (1 giây)';
      await new Promise(r => setTimeout(r, 900));

      isRecordingClip = true;
      document.getElementById('camStatusText').textContent =
        `🔴 ĐANG QUAY KÝ HIỆU (${durationSec} giây) — Hãy thực hiện động tác ngay!`;
      await new Promise(r => setTimeout(r, durationSec * 1000));
      isRecordingClip = false;

      btn.disabled = false;
      if (recordedClip76x3.length < 5) {
        document.getElementById('camStatusText').textContent =
          'Không phát hiện đủ khung hình có bàn tay. Hãy đưa tay vào vùng khung hình rồi thử lại!';
        return;
      }

      lastCapturedClip76x3 = recordedClip76x3.slice();
      document.getElementById('camStatusText').textContent =
        `Đã thu ${lastCapturedClip76x3.length} frames (76, 3). Đang chạy mô hình nhận diện...`;

      const res = await fetch('/api/webcam/predict_clip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: currentMode,
          aspect: camAspect(),
          frames_76x3: lastCapturedClip76x3
        })
      });
      const data = await res.json();
      document.getElementById('liveTelemetry').textContent = JSON.stringify(data, null, 2);
      if (data.ok) {
        document.getElementById('camStatusText').textContent =
          `Nhận diện hoàn tất trên clip Webcam (${data.captured_frames} frames)!`;
        if (currentMode === 'word' && data.word_prediction) {
          document.getElementById('liveResultContent').innerHTML =
            renderWordBadge(data.word_prediction, `Clip Webcam vừa quay (${data.captured_frames} frames)`);
        } else if (currentMode === 'sentence' && data.stream_update) {
          document.getElementById('liveResultContent').innerHTML =
            renderSentenceChips(data.stream_update.predicted_glosses, null);
        }
      } else {
        document.getElementById('camStatusText').textContent = `Lỗi: ${data.error}`;
      }
    }

    async function finetuneWithLastWebcamClip() {
      const label = document.getElementById('webcamAdaptLabelSelect').value;
      if (!label) {
        alert('Hãy chọn nhãn từ cần tinh chỉnh!');
        return;
      }
      const btn = document.getElementById('btnFinetuneCam');
      btn.disabled = true;
      document.getElementById('webcamAdaptStatus').textContent =
        `Đang lưu mẫu Webcam cho từ "${label}" và tinh chỉnh nhanh mô hình BiLSTM...`;
      try {
        const res = await fetch('/api/webcam/finetune_clip', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            label,
            aspect: camAspect(),
            frames_76x3: lastCapturedClip76x3.length >= 5 ? lastCapturedClip76x3 : null
          })
        });
        const data = await res.json();
        document.getElementById('liveTelemetry').textContent = JSON.stringify(data, null, 2);
        if (data.ok) {
          document.getElementById('webcamAdaptStatus').textContent =
            `Đã cập nhật Model với mẫu Webcam "${label}"! Dự đoán kiểm tra lại: ${data.verify_prediction.raw_top1_label} (${(data.verify_prediction.confidence * 100).toFixed(1)}%)`;
          document.getElementById('liveResultContent').innerHTML =
            renderWordBadge(data.verify_prediction, `Sau khi tinh chỉnh mẫu Webcam (${label})`);
        } else {
          document.getElementById('webcamAdaptStatus').textContent = `Lỗi: ${data.error}`;
        }
      } finally {
        btn.disabled = false;
      }
    }

    async function startWebcam() {
      const videoElement = document.getElementById('webcamVideo');
      const canvasElement = document.getElementById('webcamCanvas');
      const canvasCtx = canvasElement.getContext('2d');

      document.getElementById('camStatusText').textContent = 'Đang khởi tạo MediaPipe Holistic (Pose + 2 Hands)...';
      holisticObj = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
      });
      holisticObj.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        refineFaceLandmarks: false,
        minDetectionConfidence: 0.55,
        minTrackingConfidence: 0.55
      });

      holisticObj.onResults(async (results) => {
        const mirrorFlip = document.getElementById('chkMirrorFlip').checked;
        const { frame76, hasLeft, hasRight } = extract76x3FromHolisticResults(results, mirrorFlip);

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

        // Draw Pose upper-body shoulders (11 & 12)
        if (results.poseLandmarks && results.poseLandmarks.length > 12) {
          const p11 = results.poseLandmarks[11];
          const p12 = results.poseLandmarks[12];
          canvasCtx.strokeStyle = '#facc15';
          canvasCtx.lineWidth = 3;
          canvasCtx.beginPath();
          canvasCtx.moveTo(p11.x * canvasElement.width, p11.y * canvasElement.height);
          canvasCtx.lineTo(p12.x * canvasElement.width, p12.y * canvasElement.height);
          canvasCtx.stroke();
        }

        // Draw Left Hand (Green) and Right Hand (Cyan) according to current mirrorFlip setting
        const dispLeft = mirrorFlip ? results.rightHandLandmarks : results.leftHandLandmarks;
        const dispRight = mirrorFlip ? results.leftHandLandmarks : results.rightHandLandmarks;

        if (dispLeft) {
          drawConnectors(canvasCtx, dispLeft, HAND_CONNECTIONS, { color: '#22c55e', lineWidth: 2 });
          drawLandmarks(canvasCtx, dispLeft, { color: '#bbf7d0', lineWidth: 1, radius: 2 });
          const w = dispLeft[0];
          canvasCtx.fillStyle = '#22c55e';
          canvasCtx.font = 'bold 16px sans-serif';
          canvasCtx.fillText('L (Tay Trái)', w.x * canvasElement.width - 20, w.y * canvasElement.height - 10);
        }
        if (dispRight) {
          drawConnectors(canvasCtx, dispRight, HAND_CONNECTIONS, { color: '#38bdf8', lineWidth: 2 });
          drawLandmarks(canvasCtx, dispRight, { color: '#e0f2fe', lineWidth: 1, radius: 2 });
          const w = dispRight[0];
          canvasCtx.fillStyle = '#38bdf8';
          canvasCtx.font = 'bold 16px sans-serif';
          canvasCtx.fillText('R (Tay Phải)', w.x * canvasElement.width - 20, w.y * canvasElement.height - 10);
        }
        canvasCtx.restore();

        if (isRecordingClip && (hasLeft || hasRight)) {
          recordedClip76x3.push(frame76);
        }

        if (!isSendingFrame && !isRecordingClip) {
          isSendingFrame = true;
          try {
            const endpoint = currentMode === 'word' ? '/api/word/stream_frame' : '/api/sentence/stream_frame';
            const resp = await fetch(endpoint, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                timestamp: performance.now(),
                aspect: camAspect(),
                frame_76x3: frame76
              })
            });
            const data = await resp.json();
            if (data.ok) {
              document.getElementById('livePhaseText').textContent =
                `${data.phase} (Hands: ${data.hand_count} | Buffer: ${data.buffer_length} frames)`;
              if (currentMode === 'word' && data.word_prediction) {
                document.getElementById('liveResultContent').innerHTML = renderWordBadge(data.word_prediction, 'Luồng Webcam Trực tiếp');
              } else if (currentMode === 'sentence' && data.stream_update) {
                document.getElementById('liveResultContent').innerHTML =
                  renderSentenceChips(data.stream_update.predicted_glosses, null);
              }
              document.getElementById('liveTelemetry').textContent = JSON.stringify(data, null, 2);
            }
          } catch (e) {
          } finally {
            isSendingFrame = false;
          }
        }
      });

      cameraObj = new Camera(videoElement, {
        onFrame: async () => {
          if (holisticObj) await holisticObj.send({ image: videoElement });
        },
        width: 640,
        height: 480
      });
      await cameraObj.start();
      document.getElementById('btnStartCam').disabled = true;
      document.getElementById('btnStopCam').disabled = false;
      document.getElementById('btnRecordClip').disabled = false;
      document.getElementById('camStatusText').textContent = 'Webcam Holistic 76 điểm đang hoạt động.';
    }

    function stopWebcam() {
      if (cameraObj) {
        cameraObj.stop();
        cameraObj = null;
      }
      document.getElementById('btnStartCam').disabled = false;
      document.getElementById('btnStopCam').disabled = true;
      document.getElementById('btnRecordClip').disabled = true;
      document.getElementById('camStatusText').textContent = 'Đã tắt Webcam.';
    }

    refreshStatus().then(() => scanNpyFolder());
  </script>
</body>
</html>
"""


def _enrich_word_prediction_with_top3(
    pred: Dict[str, Any], class_names: List[str]
) -> Dict[str, Any]:
    probs = pred.get("probabilities", [])
    indexed = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)[:3]
    pred["top3"] = [
        {
            "label": class_names[idx] if idx < len(class_names) else f"CLASS_{idx}",
            "prob": float(p),
        }
        for idx, p in indexed
    ]
    return pred


def _word_pred_from_probs(
    recognizer: VSLSingleSignRecognizer, probs: np.ndarray, latency_ms: float = 0.0
) -> Dict[str, Any]:
    """Build a prediction dict (same schema as `VSLSingleSignRecognizer.predict`) from a prob vector."""
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(probs)[::-1]
    top1 = int(order[0])
    p1 = float(probs[top1])
    p2 = float(probs[order[1]]) if len(probs) > 1 else 0.0
    raw_label = recognizer.class_names[top1]
    status, label = "ACCEPTED", raw_label
    if p1 < recognizer.unknown_threshold:
        status, label = "UNKNOWN", "UNKNOWN"
    elif len(probs) > 1 and (p1 - p2) < recognizer.ambiguity_margin:
        status, label = "AMBIGUOUS", "AMBIGUOUS"
    pred = {
        "predicted_label": label,
        "raw_top1_label": raw_label,
        "confidence": p1,
        "margin": p1 - p2,
        "status": status,
        "probabilities": [float(x) for x in probs.tolist()],
        "latency_ms": float(latency_ms),
    }
    return _enrich_word_prediction_with_top3(pred, recognizer.class_names)


def _clip_to_word_features(clip_np: np.ndarray) -> List[List[float]]:
    """Webcam `(T, 76, 3)` clip -> `(30, 138)` using EXACTLY the same function as `.npy` files."""
    return npy_sequence_to_word_138_features(clip_np, target_len=SINGLE_SIGN_SEQUENCE_LENGTH)


def _predict_word_clip(clip_np: np.ndarray) -> Dict[str, Any]:
    assert STATE.word_recognizer is not None
    pred = STATE.word_recognizer.predict(_clip_to_word_features(clip_np))
    return _enrich_word_prediction_with_top3(pred, STATE.word_recognizer.class_names)


class DualVSLRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html: str, status: int = 200) -> None:
        raw = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index.html"):
            self._send_html(INDEX_HTML)
            return

        if self.path == "/api/status":
            self._send_json(
                {
                    "ok": True,
                    "default_npy_root": STATE.default_npy_root,
                    "word_model_loaded": STATE.word_recognizer is not None,
                    "word_classes": (
                        STATE.word_recognizer.class_names
                        if STATE.word_recognizer is not None
                        else []
                    ),
                    "sentence_model_loaded": STATE.sentence_recognizer is not None,
                    "sentence_num_classes": (
                        STATE.sentence_recognizer.vocab.num_classes
                        if STATE.sentence_recognizer is not None
                        else 0
                    ),
                }
            )
            return

        self._send_json({"ok": False, "error": f"Unknown endpoint {self.path}"}, status=404)

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            body = {}

        try:
            if self.path == "/api/npy/scan":
                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                info = scan_npy_dataset_root(root_dir)
                STATE.default_npy_root = root_dir
                self._send_json({"ok": True, "info": info})
                return

            # =================================================================
            # PART 1: WORD RECOGNITION (BiLSTM 30x138) FROM .NPY
            # =================================================================
            if self.path == "/api/word/train_npy":
                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                selected_folders = body.get("selected_folders")
                max_classes = int(body.get("max_classes", 8))
                max_files = int(body.get("max_files_per_class", 35))
                epochs = int(body.get("epochs", 35))
                batch_size = int(body.get("batch_size", 8))
                lr = float(body.get("lr", 1.5e-3))

                word_samples = load_word_samples_from_npy_root(
                    root_dir=root_dir,
                    selected_folders=selected_folders,
                    max_classes=max_classes,
                    max_files_per_class_split=max_files,
                )
                summary = train_single_sign_pipeline(
                    rows=word_samples,
                    output_model_path=DEFAULT_WORD_CKPT,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    seed=42,
                )
                STATE.reload_word_checkpoint(DEFAULT_WORD_CKPT, is_demo=False)
                self._send_json(
                    {
                        "ok": True,
                        "loaded_samples": len(word_samples),
                        "summary": summary,
                    }
                )
                return

            if self.path == "/api/word/predict_npy":
                STATE.ensure_word_model_loaded()
                assert STATE.word_recognizer is not None

                b64_str = str(body.get("npy_base64", ""))
                raw_bytes = base64.b64decode(b64_str)
                arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
                seq_138 = npy_sequence_to_word_138_features(arr)
                pred = STATE.word_recognizer.predict(seq_138)
                pred = _enrich_word_prediction_with_top3(
                    pred, STATE.word_recognizer.class_names
                )
                self._send_json(
                    {
                        "ok": True,
                        "filename": body.get("filename", "uploaded.npy"),
                        "trained_classes": STATE.word_recognizer.class_names,
                        "original_npy_shape": list(arr.shape),
                        "resampled_feature_shape": [len(seq_138), len(seq_138[0])],
                        "result": pred,
                    }
                )
                return

            if self.path == "/api/word/predict_random_test_npy":
                STATE.ensure_word_model_loaded()
                assert STATE.word_recognizer is not None

                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                test_dir = os.path.join(root_dir, "test")
                trained_classes = set(STATE.word_recognizer.class_names)

                # Pick a random .npy file from a folder matching one of the trained classes
                candidate_files = []
                if os.path.isdir(test_dir):
                    for folder in os.listdir(test_dir):
                        norm_f = normalize_single_gloss(folder)
                        if norm_f in trained_classes:
                            f_dir = os.path.join(test_dir, folder)
                            for fn in os.listdir(f_dir):
                                if fn.endswith(".npy"):
                                    candidate_files.append((norm_f, os.path.join(f_dir, fn)))

                if not candidate_files:
                    raise ValueError(
                        "Hãy bấm 'Huấn luyện BiLSTM Từ đơn từ thư mục .npy' trước để model học các lớp từ trong keypoints_splited."
                    )

                import random
                true_label, chosen_path = random.choice(candidate_files)
                arr = np.load(chosen_path, allow_pickle=False)
                seq_138 = npy_sequence_to_word_138_features(arr)
                pred = STATE.word_recognizer.predict(seq_138)
                pred = _enrich_word_prediction_with_top3(
                    pred, STATE.word_recognizer.class_names
                )
                self._send_json(
                    {
                        "ok": True,
                        "true_label": true_label,
                        "file_path": chosen_path,
                        "original_npy_shape": list(arr.shape),
                        "result": pred,
                    }
                )
                return

            if self.path == "/api/word/stream_frame":
                STATE.ensure_word_model_loaded()
                assert STATE.word_recognizer is not None

                frame_76x3, hand_count = parse_webcam_frame_76x3(body)
                STATE.word_step_counter += 1

                if hand_count > 0:
                    # If user had lowered hands for >= 4 frames and raises hands again, start fresh sign window
                    if STATE.word_missing_hand_streak >= 4 and len(STATE.word_live_buffer_76x3) >= 12:
                        STATE.word_live_buffer_76x3 = []
                        STATE.word_ema_probs = None
                    STATE.word_missing_hand_streak = 0
                    STATE.word_live_buffer_76x3.append(frame_76x3)
                    if len(STATE.word_live_buffer_76x3) > 36:
                        STATE.word_live_buffer_76x3 = STATE.word_live_buffer_76x3[-36:]
                else:
                    STATE.word_missing_hand_streak += 1

                buf_len = len(STATE.word_live_buffer_76x3)
                if hand_count > 0 and buf_len >= 8:
                    phase = "SIGNING"
                elif hand_count > 0:
                    phase = "SIGN_START"
                elif STATE.word_missing_hand_streak <= 3 and buf_len >= 8:
                    phase = "SIGN_END"
                else:
                    phase = "REST"

                should_infer = (
                    (hand_count > 0 and buf_len >= 8 and (STATE.word_step_counter % 2 == 0))
                    or (hand_count == 0 and STATE.word_missing_hand_streak == 2 and buf_len >= 6)
                )

                if should_infer:
                    clip_np = np.stack(STATE.word_live_buffer_76x3[-30:], axis=0)
                    STATE.last_captured_clip_76x3 = clip_np
                    raw_pred = STATE.word_recognizer.predict(_clip_to_word_features(clip_np))
                    p_vec = np.asarray(raw_pred["probabilities"], dtype=np.float64)
                    if STATE.word_ema_probs is None or len(STATE.word_ema_probs) != len(p_vec):
                        STATE.word_ema_probs = p_vec
                    else:
                        STATE.word_ema_probs = 0.55 * p_vec + 0.45 * STATE.word_ema_probs
                    STATE.last_word_prediction = _word_pred_from_probs(
                        STATE.word_recognizer, STATE.word_ema_probs, raw_pred.get("latency_ms", 0.0)
                    )

                self._send_json(
                    {
                        "ok": True,
                        "phase": phase,
                        "hand_count": hand_count,
                        "buffer_length": len(STATE.word_live_buffer_76x3),
                        "word_prediction": STATE.last_word_prediction,
                    }
                )
                return

            # =================================================================
            # PART 2: CONTINUOUS SENTENCE RECOGNITION (BiLSTM + CTC) FROM .NPY
            # =================================================================
            if self.path == "/api/sentence/train_npy":
                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                selected_folders = body.get("selected_folders")
                max_classes = int(body.get("max_classes", 6))
                sentences_per_split = int(body.get("sentences_per_split", 100))
                epochs = int(body.get("epochs", 30))
                batch_size = int(body.get("batch_size", 8))

                samples, custom_vocab = build_continuous_sentence_samples_from_npy_root(
                    root_dir=root_dir,
                    selected_folders=selected_folders,
                    max_classes=max_classes,
                    sentences_per_split=sentences_per_split,
                )
                summary = train_ctc_pipeline(
                    raw_samples=samples,
                    output_model_path=DEFAULT_SENTENCE_CKPT,
                    vocab=custom_vocab,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=1.5e-3,
                    seed=42,
                )
                export_info = export_model_to_torchscript(
                    DEFAULT_SENTENCE_CKPT, DEFAULT_SENTENCE_PT
                )
                STATE.reload_sentence_checkpoint(DEFAULT_SENTENCE_CKPT, is_demo=False)
                self._send_json(
                    {
                        "ok": True,
                        "built_sentences": len(samples),
                        "vocabulary": custom_vocab.to_dict()["words"],
                        "summary": summary,
                        "torchscript_export": export_info,
                    }
                )
                return

            if self.path == "/api/sentence/predict_npy":
                STATE.ensure_sentence_model_loaded()
                assert STATE.sentence_recognizer is not None

                decode_method = str(body.get("decode_method", "greedy"))
                files_list = body.get("files") or []
                if not files_list:
                    raise ValueError("Chưa có file .npy nào được gửi lên.")

                combined_212: List[List[float]] = []
                file_shapes = []
                for idx, f_obj in enumerate(files_list):
                    raw_bytes = base64.b64decode(str(f_obj.get("npy_base64", "")))
                    arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
                    file_shapes.append({"filename": f_obj.get("filename"), "shape": list(arr.shape)})
                    seq212 = preprocess_npy_clip_for_ctc(arr, target_clip_frames=16)
                    if idx > 0 and len(combined_212) > 0 and len(seq212) > 0:
                        prev_v = combined_212[-1]
                        next_v = seq212[0]
                        for step in range(1, 3):
                            a = step / 3.0
                            combined_212.append([prev_v[d] + a * (next_v[d] - prev_v[d]) for d in range(212)])
                    combined_212.extend(seq212)

                res = STATE.sentence_recognizer.predict_from_features(
                    combined_212, decode_method=decode_method, beam_width=5
                )
                printable = {
                    k: v
                    for k, v in res.items()
                    if k not in ("frame_probabilities", "frame_log_probabilities")
                }
                self._send_json(
                    {
                        "ok": True,
                        "uploaded_files": file_shapes,
                        "trained_vocabulary": [
                            w for k, w in sorted(STATE.sentence_recognizer.vocab.id_to_word.items()) if k != 0
                        ],
                        "total_sentence_frames": len(combined_212),
                        "result": printable,
                    }
                )
                return

            if self.path == "/api/sentence/predict_random_test_npy":
                STATE.ensure_sentence_model_loaded()
                assert STATE.sentence_recognizer is not None

                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                vocab_words = [
                    w
                    for k, w in sorted(STATE.sentence_recognizer.vocab.id_to_word.items())
                    if k != 0
                ]
                samples, _ = build_continuous_sentence_samples_from_npy_root(
                    root_dir=root_dir,
                    selected_folders=vocab_words,
                    sentences_per_split=6,
                    seed=np.random.randint(1, 10000),
                )
                test_candidates = [s for s in samples if s["speakerId"] == "spk_npy_test"] or samples
                chosen = test_candidates[0]
                feat_seq = [f["combinedFeatureVector"] for f in chosen["frames"]]

                greedy_res = STATE.sentence_recognizer.predict_from_features(
                    feat_seq, decode_method="greedy"
                )
                beam_res = STATE.sentence_recognizer.predict_from_features(
                    feat_seq, decode_method="beam_search", beam_width=5
                )

                self._send_json(
                    {
                        "ok": True,
                        "true_glosses": chosen["glossSequence"],
                        "total_frames": len(feat_seq),
                        "greedy_result": {
                            "predicted_glosses": greedy_res["predicted_glosses"],
                            "latency_ms": greedy_res["latency_ms"],
                        },
                        "beam_search_result": {
                            "predicted_glosses": beam_res["predicted_glosses"],
                            "latency_ms": beam_res["latency_ms"],
                        },
                    }
                )
                return

            if self.path == "/api/sentence/stream_frame":
                STATE.ensure_sentence_model_loaded()
                assert STATE.sentence_recognizer is not None

                frame_76x3, hand_count = parse_webcam_frame_76x3(body)
                STATE.sentence_step_counter += 1

                if hand_count > 0:
                    if STATE.sentence_missing_hand_streak >= 10 and len(STATE.sentence_live_buffer_76x3) >= 40:
                        STATE.sentence_live_buffer_76x3 = []
                    STATE.sentence_missing_hand_streak = 0
                    STATE.sentence_live_buffer_76x3.append(frame_76x3)
                    if len(STATE.sentence_live_buffer_76x3) > 96:
                        STATE.sentence_live_buffer_76x3 = STATE.sentence_live_buffer_76x3[-96:]
                else:
                    STATE.sentence_missing_hand_streak += 1

                buf_len = len(STATE.sentence_live_buffer_76x3)
                phase = "SIGNING" if hand_count > 0 else "REST"

                if buf_len >= 12 and (
                    (hand_count > 0 and STATE.sentence_step_counter % 4 == 0)
                    or (hand_count == 0 and STATE.sentence_missing_hand_streak == 2)
                ):
                    clip_np = np.stack(STATE.sentence_live_buffer_76x3, axis=0)
                    STATE.last_captured_clip_76x3 = clip_np
                    target_ctc_len = max(16, min(64, int(round(buf_len * 0.65))))
                    seq212 = preprocess_npy_clip_for_ctc(clip_np, target_clip_frames=target_ctc_len)
                    res = STATE.sentence_recognizer.predict_from_features(
                        seq212, decode_method="beam_search", beam_width=5
                    )
                    STATE.last_sentence_prediction = {
                        "predicted_glosses": res.get("predicted_glosses", []),
                        "predicted_sentence": " ".join(res.get("predicted_glosses", [])),
                        "confidence": float(res.get("confidence", 0.0)),
                        "latency_ms": float(res.get("latency_ms", 0.0)),
                    }

                self._send_json(
                    {
                        "ok": True,
                        "phase": phase,
                        "hand_count": hand_count,
                        "buffer_length": buf_len,
                        "stream_update": STATE.last_sentence_prediction,
                    }
                )
                return

            if self.path == "/api/webcam/predict_clip":
                mode = str(body.get("mode", "word"))
                clip_np = parse_webcam_clip_76x3(body)
                if clip_np is None:
                    raise ValueError("Cần ít nhất 4 frame có bàn tay để nhận diện.")
                STATE.last_captured_clip_76x3 = clip_np

                if mode == "word":
                    STATE.ensure_word_model_loaded()
                    assert STATE.word_recognizer is not None
                    pred = _predict_word_clip(clip_np)
                    STATE.last_word_prediction = pred
                    self._send_json(
                        {
                            "ok": True,
                            "mode": "word",
                            "captured_frames": int(clip_np.shape[0]),
                            "word_prediction": pred,
                        }
                    )
                    return
                else:
                    STATE.ensure_sentence_model_loaded()
                    assert STATE.sentence_recognizer is not None
                    target_len = max(16, min(64, int(round(clip_np.shape[0] * 0.65))))
                    seq212 = preprocess_npy_clip_for_ctc(clip_np, target_clip_frames=target_len)
                    res = STATE.sentence_recognizer.predict_from_features(
                        seq212, decode_method="beam_search", beam_width=5
                    )
                    stream_update = {
                        "predicted_glosses": res.get("predicted_glosses", []),
                        "predicted_sentence": " ".join(res.get("predicted_glosses", [])),
                        "confidence": float(res.get("confidence", 0.0)),
                        "latency_ms": float(res.get("latency_ms", 0.0)),
                    }
                    STATE.last_sentence_prediction = stream_update
                    self._send_json(
                        {
                            "ok": True,
                            "mode": "sentence",
                            "captured_frames": int(clip_np.shape[0]),
                            "stream_update": stream_update,
                        }
                    )
                    return

            if self.path == "/api/webcam/finetune_clip":
                STATE.ensure_word_model_loaded()
                assert STATE.word_recognizer is not None

                label = normalize_single_gloss(str(body.get("label", "")))
                if not label:
                    raise ValueError("Vui lòng chọn nhãn từ để tinh chỉnh.")
                if label not in STATE.word_recognizer.class_names:
                    raise ValueError(
                        f"Nhãn '{label}' không nằm trong danh sách lớp hiện tại: {STATE.word_recognizer.class_names}"
                    )

                clip_np = parse_webcam_clip_76x3(body)
                if clip_np is not None:
                    pass
                elif STATE.last_captured_clip_76x3 is not None and len(STATE.last_captured_clip_76x3) >= 4:
                    clip_np = STATE.last_captured_clip_76x3
                elif len(STATE.word_live_buffer_76x3) >= 4:
                    clip_np = np.stack(STATE.word_live_buffer_76x3[-30:], axis=0)
                else:
                    raise ValueError("Chưa có dữ liệu Webcam vừa quay để lưu & tinh chỉnh.")

                clip_np = np.nan_to_num(clip_np, nan=0.0, posinf=0.0, neginf=0.0)
                # Save `.npy` file to `data/webcam_npy/<label>/`
                label_dir = os.path.join(WEBCAM_NPY_DIR, label)
                os.makedirs(label_dir, exist_ok=True)
                save_path = os.path.join(label_dir, f"webcam_{int(time.time() * 1000)}.npy")
                np.save(save_path, clip_np)

                # Fast online adaptation on the active BiLSTM word model
                import random as _random

                class_names = STATE.word_recognizer.class_names
                rng = _random.Random(int(time.time()))
                xs: List[List[List[float]]] = []
                ys: List[int] = []

                # (a) ALL webcam clips saved so far for the trained classes (incl. the one just saved)
                seen_hashes = set()
                for cls in class_names:
                    cls_dir = os.path.join(WEBCAM_NPY_DIR, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    for fn in os.listdir(cls_dir):
                        if not fn.endswith(".npy"):
                            continue
                        try:
                            w_arr = np.load(os.path.join(cls_dir, fn), allow_pickle=False)
                        except Exception:
                            continue
                        h = hash(w_arr.tobytes())
                        if h in seen_hashes:  # skip duplicate saves of the same clip
                            continue
                        seen_hashes.add(h)
                        n_aug = 10 if os.path.join(cls_dir, fn) == save_path else 4
                        xs.append(_clip_to_word_features(w_arr))
                        ys.append(class_names.index(cls))
                        for _ in range(n_aug):
                            xs.append(_clip_to_word_features(augment_npy_76x3(w_arr, rng)))
                            ys.append(class_names.index(cls))

                # (b) Replay anchors from the original `.npy` dataset to prevent forgetting (cached)
                if STATE.word_replay_cache is None:
                    STATE.word_replay_cache = []
                    if os.path.isdir(STATE.default_npy_root):
                        try:
                            replay = load_word_samples_from_npy_root(
                                root_dir=STATE.default_npy_root,
                                selected_folders=class_names,
                                max_classes=len(class_names),
                                max_files_per_class_split=40,
                                min_train_samples_per_class=0,
                                seed=99,
                            )
                            STATE.word_replay_cache = [
                                (s["sequence"], class_names.index(s["label"]))
                                for s in replay
                                if s.get("split") == "train" and s["label"] in class_names
                            ]
                        except Exception as e:
                            print(f"[FINETUNE] replay load failed: {e}")
                for seq, y in STATE.word_replay_cache:
                    xs.append(seq)
                    ys.append(y)

                device = STATE.word_recognizer.device
                model = STATE.word_recognizer.model
                x_tensor = torch.tensor(np.asarray(xs, dtype=np.float32), device=device)
                y_tensor = torch.tensor(ys, dtype=torch.long, device=device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
                criterion = torch.nn.CrossEntropyLoss()
                model.train()
                n = x_tensor.shape[0]
                for _ in range(12):
                    perm = torch.randperm(n, device=device)
                    for b in range(0, n, 16):
                        idx = perm[b : b + 16]
                        if idx.numel() < 2:
                            continue
                        optimizer.zero_grad()
                        loss = criterion(model.forward_logits(x_tensor[idx]), y_tensor[idx])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        optimizer.step()
                model.eval()

                # Save updated checkpoint
                ckpt = torch.load(DEFAULT_WORD_CKPT, map_location="cpu", weights_only=False)
                ckpt["model_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(ckpt, DEFAULT_WORD_CKPT)

                pred = _predict_word_clip(clip_np)
                STATE.last_word_prediction = pred
                STATE.word_ema_probs = None

                self._send_json(
                    {
                        "ok": True,
                        "saved_npy_path": save_path,
                        "adapted_label": label,
                        "finetune_samples": int(n),
                        "verify_prediction": pred,
                    }
                )
                return

            if self.path == "/api/stream_reset":
                STATE.reset_word_stream()
                STATE.reset_sentence_stream()
                self._send_json({"ok": True})
                return

            self._send_json({"ok": False, "error": f"Unknown POST path {self.path}"}, status=404)
        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                status=400,
            )


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), DualVSLRequestHandler)
    print(f"VSL Studio (.npy keypoints_splited enabled) running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
