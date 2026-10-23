# VSL Continuous Recognition Model

An independent, trainable PyTorch machine learning repository for **Vietnamese Sign Language (VSL) Continuous Gloss Sequence Recognition** (`VSLSignCTCModel`) and **Single-Sign Classification Baseline** (`VSLSingleSignBaselineModel`), built from a comprehensive audit of the `vsl-hand-landmarks-studio` source code.

---

## 1. Repository Structure

```text
vsl_continuous_recognition_model/
├── config.yaml                 # Hyperparameters, architecture, streaming, and boundary detector config
├── ctc_vocabulary.json         # Canonical 21-gloss Vietnamese Sign Language vocabulary (<blank> = 0)
├── requirements.txt            # Python package dependencies
├── feature_extractor.py        # 106-D / 138-D invariant features, 212-D canonical two-hand slot extraction,
│                               # <=500ms occlusion interpolation, and 3-phase SignBoundaryDetector
├── decoder.py                  # Canonical NFC gloss normalization, greedy & prefix beam search CTC decoders,
│                               # and Levenshtein WER / CER / S-D-I-H edit-distance metrics
├── model.py                    # VSLSignCTCModel (Conv1D + 2-layer BiLSTM + Linear + LogSoftmax) &
│                               # VSLSingleSignBaselineModel (30x138 Conv1D + MaxPool1D + BiLSTM + Dense)
├── dataset.py                  # Continuous & single-sign dataset loaders, P0-1..P0-7 validation rules,
│                               # NaN/Inf sanitization, and leak-free group splitting
├── train.py                    # Reproducible training pipeline with AdamW, cosine schedule, grad clipping,
│                               # validation WER/loss checkpointing, and early stopping
├── evaluate.py                 # Standalone evaluation script (WER, CER, Exact Match, S/D/I/H, per-speaker,
│                               # sequence length & gloss count buckets, latency, and boundary evaluation)
├── inference.py                # Standalone VSLContinuousRecognizer, rolling-window CTCStreamInferenceEngine,
│                               # VSLSingleSignRecognizer, and TorchScript (.pt) export utility
├── tests/
│   └── test_all.py             # 12-part unit & integration test suite (including 1e-5 numerical parity check)
└── examples/
    └── demo_inference.py       # Runnable end-to-end training, export, streaming, and inference demonstration
```

---

## 2. Architecture & Compatibility Contract

### Primary Deliverable: `VSLSignCTCModel` (`model.py`)
- **Input Tensor**: `[batch_size, sequence_length, 212]` (`float32`, variable length $T$)
- **Feature Contract**: `feature_version = "v1-canonical-left106-right106"` (`212 = Left 106 + Right 106`)
  - Indices `0..105`: Canonical **Left** hand 106-D invariant vector (`63` normalized coordinates + `5` fingertip distances + `5` finger curl angles + `10` adjacent inter-finger distances + `2` palm normal X/Y + `21` temporal velocities), or `106` zeros when the left hand is absent.
  - Indices `106..211`: Canonical **Right** hand 106-D invariant vector, or `106` zeros when the right hand is absent.
- **Temporal Convolution Encoder**:
  - `Conv1d(212 -> 64, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm1d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)`
  - Explicit validity masking (`x = x * mask`, `x_conv = x_conv * mask`) before and after `Conv1d` prevents padded frames from leaking into valid time steps via the size-3 convolution kernel.
- **Bidirectional Recurrent Encoder**:
  - `LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)` $\rightarrow$ `256` features per frame.
  - Uses `pack_padded_sequence(..., enforce_sorted=False)` and `pad_packed_sequence(..., total_length=max_t)` when sequence lengths are provided, guaranteeing exact mathematical equivalence (`atol=1e-5`) between single-sequence and padded-batch inference.
- **Classification Head**:
  - `Linear(256 -> 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(64 -> 22)` $\rightarrow$ `LogSoftmax(dim=-1)`
- **Output Tensor**: `[batch_size, sequence_length, 22]` log-probabilities (`<blank>` at index `0` + 21 canonical NFC-normalized VSL glosses at indices `1..21`).

### Secondary Deliverable: `VSLSingleSignBaselineModel` (`model.py`)
- **Input Tensor**: `[batch_size, 30, 138]` (`106` manual features + `32` Non-Manual Markers)
- **Layers**: `Conv1d(138 -> 32, kernel_size=3)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)` $\rightarrow$ `BiLSTM(32 -> 32/dir)` (last step `64`-D) $\rightarrow$ `Dropout(0.25)` $\rightarrow$ `Linear(64 -> 32)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(32 -> num_classes)`.
- **Rejection Gates**: `UNKNOWN` when $\max_c P(c) < 0.55$; `AMBIGUOUS` when $P_{(1)} - P_{(2)} < 0.12$.

---

## 3. Setup & Installation

```powershell
pip install -r requirements.txt
```

---

## 4. Running Tests & End-to-End Verification

Run the 12-part unit and integration test suite (verifies $10^{-5}$ numerical parity against `parity_fixture.json`, 212-D canonical hand slot invariance, occlusion interpolation, NFC gloss normalization, CTC minimum-length validation, greedy & beam-search decoding, group-disjoint splitting, batch-vs-single inference invariance, training/checkpoint round-trip, and boundary detection):

```powershell
python tests/test_all.py
```

Run the end-to-end demonstration script:

```powershell
python examples/demo_inference.py
```

---

## 5. Training, Evaluation & Standalone Inference Commands

### Train Continuous CTC Model
```powershell
python train.py `
  --config config.yaml `
  --dataset_path path/to/continuous_vsl_dataset.json `
  --output_model checkpoints/vsl_ctc_best.pth `
  --mode ctc `
  --epochs 80 `
  --batch_size 16 `
  --lr 0.001 `
  --weight_decay 0.0001 `
  --grad_clip 2.0 `
  --seed 42
```

### Train Single-Sign Baseline Model
```powershell
python train.py `
  --dataset_path path/to/single_sign_dataset.json `
  --output_model checkpoints/vsl_single_sign_best.pth `
  --mode single_sign `
  --epochs 60 `
  --batch_size 16 `
  --seed 42
```

### Evaluate a Saved Checkpoint
```powershell
python evaluate.py `
  --checkpoint checkpoints/vsl_ctc_best.pth `
  --dataset_path path/to/continuous_vsl_dataset.json `
  --mode ctc `
  --split test `
  --decode_method beam_search `
  --beam_width 5 `
  --output_report reports/test_eval_report.json
```

### Run Standalone Inference & Export to TorchScript
```powershell
python inference.py `
  --checkpoint checkpoints/vsl_ctc_best.pth `
  --input_json path/to/sample_sequence.json `
  --decode_method greedy `
  --export_torchscript checkpoints/vsl_ctc_best.pt
```

---

## 6. Scientific Disclosure & Dataset Limitations

1. **No Bundled Real-World Multi-Speaker Dataset in Source Archive**:
   - Inspection of the supplied repository (`vsl-hand-landmarks-studio (43)`) confirmed that user recordings in the web application are stored dynamically inside browser `IndexedDB` (`src/utils/ctcDataset.ts`) or exported manually via JSON. Only a 4-frame single-hand numerical parity fixture (`parity_fixture.json` and `python_output.json`) and the 21-word vocabulary (`src/data/ctc_vocabulary.json`) were present on disk.
2. **Distinction Between Pipeline Verification and Real-World Generalization**:
   - All training loops, CTC loss computations, group-disjoint splitters, decoders, evaluation metrics, and export routines in this repository have been executed and verified via automated unit tests (`tests/test_all.py`) and pipeline smoke tests (`examples/demo_inference.py`).
   - However, **no real-world VSL recognition accuracy or generalization WER is claimed**, because training a production sign language recognizer requires a real multi-signer corpus containing at least $\ge 3$ independent signers (`speakerId`) recorded across varied lighting conditions and signing speeds.
