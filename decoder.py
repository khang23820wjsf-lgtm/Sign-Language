#!/usr/bin/env python3
"""
===============================================================================
CANONICAL VOCABULARY, CTC DECODING (GREEDY + BEAM SEARCH) & METRICS
Exact reproduction of the VSL canonical gloss vocabulary normalization,
token ID mapping, CTC greedy & prefix beam search decoders, and edit-distance
evaluation metrics (WER, CER, S/D/I/H, Exact Sequence Match).
===============================================================================
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

BLANK_ID: int = 0
BLANK_TOKEN: str = "<blank>"
FEATURE_DIM: int = 212
FEATURE_VERSION: str = "v1-canonical-left106-right106"

DEFAULT_CANONICAL_WORDS: List[str] = [
    "BÁC_SĨ", "BẠN", "BỤNG", "BỆNH_VIỆN", "CẦN", "CẢM_ƠN", "GIÚP_ĐỠ",
    "GÌ", "KHÔNG", "KHOẺ", "NHÀ", "NƯỚC", "SỐT", "THUỐC", "TÊN",
    "TÔI", "UỐNG", "XIN_CHÀO", "ĂN", "ĐAU", "ĐẦU",
]


def normalize_single_gloss(raw: Optional[str]) -> str:
    """
    Canonical gloss normalization (P0-7):
    1. Trim whitespace
    2. Normalize Unicode NFC
    3. Uppercase conversion
    4. Replace consecutive spaces, hyphens, or underscores with a single '_'
    5. Strip leading/trailing underscores
    """
    if not raw:
        return ""
    s = unicodedata.normalize("NFC", str(raw).strip())
    s = s.upper()
    s = re.sub(r"[\s\-_]+", "_", s)
    s = s.strip("_")
    return s


def canonicalize_gloss_sequence(raw_input: Union[str, Sequence[str], None]) -> List[str]:
    """Parse and normalize a raw gloss string or list into canonical gloss tokens."""
    if not raw_input:
        return []
    if isinstance(raw_input, (list, tuple)):
        return [normalize_single_gloss(g) for g in raw_input if normalize_single_gloss(g)]
    trimmed = str(raw_input).strip()
    if not trimmed:
        return []
    tokens = trimmed.split(",") if "," in trimmed else re.split(r"\s+", trimmed)
    return [normalize_single_gloss(t) for t in tokens if normalize_single_gloss(t)]


def load_canonical_words(vocab_path: Optional[str] = None) -> List[str]:
    """Load canonical gloss words from ctc_vocabulary.json if available."""
    possible_paths: List[str] = []
    if vocab_path:
        possible_paths.append(vocab_path)
    possible_paths.extend([
        os.path.join(os.path.dirname(__file__), "ctc_vocabulary.json"),
        os.path.join(os.getcwd(), "ctc_vocabulary.json"),
        os.path.join(os.path.dirname(__file__), "..", "src", "data", "ctc_vocabulary.json"),
    ])
    for p in possible_paths:
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    words = data.get("words", [])
                    if words:
                        return [normalize_single_gloss(w) for w in words if normalize_single_gloss(w)]
            except Exception as e:
                print(f"[VOCAB] Warning: could not read {p}: {e}")
    return list(DEFAULT_CANONICAL_WORDS)


class Vocabulary:
    """
    Deterministic bidirectional mapping between canonical VSL glosses and integer IDs.
    ID 0 is strictly reserved for `<blank>`.
    IDs 1..N are assigned to sorted unique normalized glosses.
    """

    def __init__(self, custom_words: Optional[Sequence[str]] = None, vocab_path: Optional[str] = None) -> None:
        words = list(custom_words) if custom_words else load_canonical_words(vocab_path)
        unique = sorted(list({normalize_single_gloss(w) for w in words if normalize_single_gloss(w)}))
        self.id_to_word: Dict[int, str] = {BLANK_ID: BLANK_TOKEN}
        self.word_to_id: Dict[str, int] = {BLANK_TOKEN: BLANK_ID}
        for idx, w in enumerate(unique, start=1):
            self.id_to_word[idx] = w
            self.word_to_id[w] = idx

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_word) - 1

    @property
    def num_classes(self) -> int:
        return len(self.id_to_word)

    def get_gloss(self, token_id: int) -> Optional[str]:
        return self.id_to_word.get(int(token_id))

    def get_id(self, gloss: str) -> Optional[int]:
        norm = normalize_single_gloss(gloss)
        return self.word_to_id.get(norm)

    def has_gloss(self, gloss: str) -> bool:
        norm = normalize_single_gloss(gloss)
        return norm in self.word_to_id and self.word_to_id[norm] != BLANK_ID

    def encode(self, glosses: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for g in glosses:
            norm = normalize_single_gloss(g)
            if norm in self.word_to_id and self.word_to_id[norm] != BLANK_ID:
                ids.append(self.word_to_id[norm])
            else:
                raise ValueError(f"Gloss '{g}' (normalized: '{norm}') is not in vocabulary.")
        return ids

    def validate_and_tokenize(self, gloss_sequence: Sequence[str]) -> Tuple[bool, List[int], List[str]]:
        token_ids: List[int] = []
        unknown_glosses: List[str] = []
        for raw_gloss in gloss_sequence:
            norm = normalize_single_gloss(raw_gloss)
            if not norm:
                continue
            tid = self.word_to_id.get(norm)
            if tid is not None and tid != BLANK_ID:
                token_ids.append(tid)
            else:
                unknown_glosses.append(norm)
        is_valid = len(unknown_glosses) == 0 and len(token_ids) > 0
        return is_valid, token_ids, unknown_glosses

    def decode(self, token_ids: Sequence[int]) -> List[str]:
        return [self.id_to_word.get(int(i), f"UNK_{i}") for i in token_ids if int(i) != BLANK_ID]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blankId": BLANK_ID,
            "blankToken": BLANK_TOKEN,
            "vocabSize": self.vocab_size,
            "numClasses": self.num_classes,
            "idToWord": {int(k): v for k, v in sorted(self.id_to_word.items())},
            "words": [v for k, v in sorted(self.id_to_word.items()) if k != BLANK_ID],
        }

    @classmethod
    def from_checkpoint_vocab(cls, vocab_data: Union[Dict[Any, str], Sequence[str]]) -> "Vocabulary":
        if isinstance(vocab_data, dict):
            inst = cls.__new__(cls)
            inst.id_to_word = {}
            inst.word_to_id = {}
            for k, v in vocab_data.items():
                ik = int(k)
                sv = str(v) if ik == BLANK_ID else normalize_single_gloss(str(v))
                inst.id_to_word[ik] = sv
                inst.word_to_id[sv] = ik
            if BLANK_ID not in inst.id_to_word:
                inst.id_to_word[BLANK_ID] = BLANK_TOKEN
                inst.word_to_id[BLANK_TOKEN] = BLANK_ID
            return inst
        if isinstance(vocab_data, (list, tuple)):
            words = [w for w in vocab_data if w != BLANK_TOKEN]
            return cls(custom_words=words)
        return cls()


# =============================================================================
# 2. CTC DECODING (GREEDY & PREFIX BEAM SEARCH)
# =============================================================================

@dataclass
class CTCGreedyDecodeResult:
    raw_token_ids: List[int]
    collapsed_token_ids: List[int]
    decoded_token_ids: List[int]
    glosses: List[str]


def _extract_2d_frame_scores(log_probs_or_probs: Any) -> Optional[List[List[float]]]:
    """Convert tensor or nested list `[T, C]` into list of float lists, or None if 1D token IDs."""
    if hasattr(log_probs_or_probs, "detach"):
        arr = log_probs_or_probs.detach().cpu().tolist()
    elif hasattr(log_probs_or_probs, "tolist"):
        arr = log_probs_or_probs.tolist()
    else:
        arr = list(log_probs_or_probs)

    if not arr:
        return []
    if isinstance(arr[0], (list, tuple)):
        return [[float(x) for x in row] for row in arr]
    return None


def decode_ctc_greedy_detailed(
    log_probs_seq: Any,
    vocab: Union[Vocabulary, Dict[int, str], Callable[[int], Optional[str]]],
) -> CTCGreedyDecodeResult:
    """
    Greedy CTC decoding with full alignment trace:
    1. Select argmax token at each time step t.
    2. Collapse consecutive repeated token IDs: [A, A, blank, B, B] -> [A, blank, B].
    3. Remove blank tokens (ID = 0).
    4. Preserve repeated glosses separated by blank tokens: [A, blank, A] -> [A, A].
    """
    if hasattr(log_probs_seq, "detach"):
        raw_token_ids = [int(i) for i in log_probs_seq.argmax(dim=-1).tolist()]
    elif hasattr(log_probs_seq, "argmax") and hasattr(log_probs_seq, "ndim"):
        if log_probs_seq.ndim >= 2:
            raw_token_ids = [int(i) for i in log_probs_seq.argmax(axis=-1).tolist()]
        else:
            raw_token_ids = [int(i) for i in log_probs_seq.tolist()]
    else:
        frames_2d = _extract_2d_frame_scores(log_probs_seq)
        if frames_2d is not None:
            raw_token_ids = [
                max(range(len(frame)), key=lambda c: frame[c]) if frame else 0
                for frame in frames_2d
            ]
        else:
            raw_token_ids = [int(t) for t in log_probs_seq]

    collapsed_token_ids: List[int] = []
    for i, token in enumerate(raw_token_ids):
        if i == 0 or token != raw_token_ids[i - 1]:
            collapsed_token_ids.append(token)

    decoded_token_ids = [t for t in collapsed_token_ids if t != BLANK_ID]

    def resolve(tid: int) -> str:
        if isinstance(vocab, Vocabulary):
            return vocab.id_to_word.get(tid, f"TOKEN_{tid}")
        if callable(vocab):
            return vocab(tid) or f"TOKEN_{tid}"
        return vocab.get(tid, f"TOKEN_{tid}")

    glosses = [resolve(tid) for tid in decoded_token_ids]
    return CTCGreedyDecodeResult(
        raw_token_ids=raw_token_ids,
        collapsed_token_ids=collapsed_token_ids,
        decoded_token_ids=decoded_token_ids,
        glosses=glosses,
    )


def decode_ctc_greedy(
    log_probs_seq: Any,
    vocab: Union[Vocabulary, Dict[int, str], Callable[[int], Optional[str]]],
) -> List[str]:
    """Standard Greedy CTC decoder returning the decoded gloss list."""
    return decode_ctc_greedy_detailed(log_probs_seq, vocab).glosses


def decode_ctc_beam_search(
    scores_seq: Any,
    vocab: Union[Vocabulary, Dict[int, str], Callable[[int], Optional[str]]],
    beam_width: int = 10,
    is_log_probs: Optional[bool] = None,
) -> Tuple[List[str], List[int]]:
    """
    Prefix Beam Search CTC Decoder (matching `decodeCTCBeamSearch` in `src/utils/ctcDecoder.ts`).
    Accepts either log-probabilities `[T, C]` (from PyTorch `log_softmax`) or probabilities `[T, C]`.
    Returns `(decoded_glosses, decoded_token_ids)`.
    """
    frames = _extract_2d_frame_scores(scores_seq)
    if not frames:
        return [], []

    # Auto-detect whether inputs are already log-probabilities (values <= 0 with row exp sum ~ 1)
    if is_log_probs is None:
        first_row = frames[0]
        is_log_probs = any(v < 0.0 for v in first_row)

    neg_inf = -1e9

    def log_add(a: float, b: float) -> float:
        if a <= neg_inf:
            return b
        if b <= neg_inf:
            return a
        return max(a, b) + math.log1p(math.exp(-abs(a - b)))

    # Map: prefix tuple -> [p_blank, p_non_blank]
    beam: Dict[Tuple[int, ...], List[float]] = {(): [0.0, neg_inf]}

    for frame in frames:
        if is_log_probs:
            log_frame = [max(float(v), -30.0) for v in frame]
        else:
            log_frame = [math.log(max(float(v), 1e-12)) for v in frame]

        next_beam: Dict[Tuple[int, ...], List[float]] = {}

        def get_entry(prefix: Tuple[int, ...]) -> List[float]:
            if prefix not in next_beam:
                next_beam[prefix] = [neg_inf, neg_inf]
            return next_beam[prefix]

        for prefix, (p_b, p_nb) in beam.items():
            p_total = log_add(p_b, p_nb)

            # 1. Extend with BLANK
            blank_prob = log_frame[BLANK_ID]
            b_entry = get_entry(prefix)
            b_entry[0] = log_add(b_entry[0], p_total + blank_prob)

            # 2. Extend with non-blank token c
            last_token = prefix[-1] if prefix else None
            for c in range(1, len(log_frame)):
                c_prob = log_frame[c]
                if c == last_token:
                    # Repeated token requires previous frame to be blank to emit a second token
                    new_entry = get_entry(prefix + (c,))
                    new_entry[1] = log_add(new_entry[1], p_b + c_prob)

                    # Otherwise collapses into current prefix
                    same_entry = get_entry(prefix)
                    same_entry[1] = log_add(same_entry[1], p_nb + c_prob)
                else:
                    new_entry = get_entry(prefix + (c,))
                    new_entry[1] = log_add(new_entry[1], p_total + c_prob)

        # Prune to top beam_width
        sorted_entries = sorted(
            next_beam.items(),
            key=lambda kv: log_add(kv[1][0], kv[1][1]),
            reverse=True,
        )[: max(1, beam_width)]
        beam = dict(sorted_entries)

    best_prefix, _ = max(beam.items(), key=lambda kv: log_add(kv[1][0], kv[1][1]))
    token_ids = list(best_prefix)

    def resolve(tid: int) -> str:
        if isinstance(vocab, Vocabulary):
            return vocab.id_to_word.get(tid, f"TOKEN_{tid}")
        if callable(vocab):
            return vocab(tid) or f"TOKEN_{tid}"
        return vocab.get(tid, f"TOKEN_{tid}")

    return [resolve(tid) for tid in token_ids], token_ids


# =============================================================================
# 3. EDIT-DISTANCE METRICS (WER, CER, S/D/I/H, SEQUENCE EXACT MATCH)
# =============================================================================

@dataclass
class EditDistanceDetail:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    edit_distance: int
    reference_length: int
    hypothesis_length: int


def compute_edit_distance_detail(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> EditDistanceDetail:
    """
    Token-level Levenshtein dynamic programming with backtracked S, D, I, H counts.
    Exact reproduction of `computeGlossErrorRate` in `src/utils/ctcDecoder.ts`.
    """
    ref = list(reference or [])
    hyp = list(hypothesis or [])
    n = len(ref)
    m = len(hyp)

    if n == 0:
        return EditDistanceDetail(
            wer=0.0 if m == 0 else 1.0,
            substitutions=0,
            deletions=0,
            insertions=m,
            hits=0,
            edit_distance=m,
            reference_length=0,
            hypothesis_length=m,
        )

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # Substitution
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                )

    i, j = n, m
    substitutions = 0
    deletions = 0
    insertions = 0
    hits = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    edit_dist = substitutions + deletions + insertions
    wer_val = edit_dist / n
    return EditDistanceDetail(
        wer=wer_val,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        hits=hits,
        edit_distance=edit_dist,
        reference_length=n,
        hypothesis_length=m,
    )


def compute_wer(ref: Sequence[str], hyp: Sequence[str]) -> float:
    """Gloss Error Rate / Word Error Rate via Levenshtein dynamic programming."""
    return compute_edit_distance_detail(ref, hyp).wer


def compute_cer(ref: Sequence[str], hyp: Sequence[str]) -> float:
    """Character Error Rate (CER) across joined gloss strings."""
    ref_chars = list(" ".join(ref)) if ref else []
    hyp_chars = list(" ".join(hyp)) if hyp else []
    return compute_edit_distance_detail(ref_chars, hyp_chars).wer


def compute_sequence_exact_match(ref: Sequence[str], hyp: Sequence[str]) -> bool:
    """Exact sequence match accuracy check."""
    return list(ref) == list(hyp)
