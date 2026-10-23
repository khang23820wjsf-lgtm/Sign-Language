"""
Train and verify both models (Part 1: Word BiLSTM & Part 2: Sentence BiLSTM + CTC)
on real `.npy` files from `keypoints_splited` and save ready-to-use checkpoints
to `checkpoints/vsl_word_bilstm.pth` and `checkpoints/vsl_sentence_ctc.pth`.
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from inference import VSLContinuousRecognizer, VSLSingleSignRecognizer, export_model_to_torchscript
from train import train_ctc_pipeline, train_single_sign_pipeline
from vsl400_npy_loader import (
    build_continuous_sentence_samples_from_npy_root,
    find_default_npy_root,
    load_word_samples_from_npy_root,
)


def main() -> None:
    root_dir = find_default_npy_root()
    if not root_dir:
        raise FileNotFoundError("Could not find default keypoints_splited directory.")

    os.makedirs("checkpoints", exist_ok=True)
    word_ckpt = os.path.join("checkpoints", "vsl_word_bilstm.pth")
    sent_ckpt = os.path.join("checkpoints", "vsl_sentence_ctc.pth")
    sent_pt = os.path.join("checkpoints", "vsl_sentence_ctc.pt")

    print(f"[1/2] Loading & training Part 1 (Word BiLSTM) from: {root_dir}")
    word_samples = load_word_samples_from_npy_root(
        root_dir=root_dir,
        max_classes=8,
        max_files_per_class_split=35,
        min_train_samples_per_class=28,
        seed=42,
    )
    word_res = train_single_sign_pipeline(
        rows=word_samples,
        output_model_path=word_ckpt,
        epochs=35,
        batch_size=8,
        lr=1.5e-3,
        seed=42,
    )
    print(
        f"  -> Word BiLSTM Trained Classes ({len(word_res['class_names'])}): {word_res['class_names']}\n"
        f"  -> Best Val Accuracy : {word_res['best_val_accuracy']:.2%}\n"
        f"  -> Held-out Test Acc : {word_res['test_metrics']['accuracy']:.2%}\n"
        f"  -> Held-out Macro-F1 : {word_res['test_metrics']['macro_f1']:.2%}"
    )

    # Verify standalone VSLSingleSignRecognizer on held-out test samples
    word_rec = VSLSingleSignRecognizer(word_ckpt)
    test_rows = [r for r in word_samples if r.get("split") == "test"]
    accepted_hits = 0
    top1_hits = 0
    for r in test_rows:
        p = word_rec.predict(r["sequence"])
        if p["predicted_label"] == r["label"]:
            accepted_hits += 1
        if p["raw_top1_label"] == r["label"]:
            top1_hits += 1
    print(
        f"  -> VSLSingleSignRecognizer Held-Out Test Verification ({len(test_rows)} files): "
        f"Top-1={top1_hits}/{len(test_rows)} ({top1_hits / max(1, len(test_rows)):.2%}), "
        f"Accepted={accepted_hits}/{len(test_rows)} ({accepted_hits / max(1, len(test_rows)):.2%})"
    )

    print(f"\n[2/2] Building & training Part 2 (Sentence 2-Layer BiLSTM + CTC) from: {root_dir}")
    ctc_samples, custom_vocab = build_continuous_sentence_samples_from_npy_root(
        root_dir=root_dir,
        max_classes=6,
        sentences_per_split=100,
        seed=42,
    )
    ctc_res = train_ctc_pipeline(
        raw_samples=ctc_samples,
        output_model_path=sent_ckpt,
        vocab=custom_vocab,
        epochs=30,
        batch_size=8,
        lr=1.5e-3,
        seed=42,
    )
    export_model_to_torchscript(sent_ckpt, sent_pt)
    print(
        f"  -> Sentence CTC Vocabulary ({custom_vocab.num_classes - 1} words): {custom_vocab.to_dict()['words']}\n"
        f"  -> Best Val WER        : {ctc_res['best_val_wer']:.2%}\n"
        f"  -> Held-out Test WER   : {ctc_res['test_metrics']['corpus_wer']:.2%}\n"
        f"  -> Held-out Exact Match: {ctc_res['test_metrics']['exact_match_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
