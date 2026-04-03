# Device Fingerprinting Tools

This subdirectory contains all source code for training TripletNet-based device
fingerprint extractors on AODT PUSCH IQ data from Hugging Face, evaluating
closed-set and open-set (enrollment-based) authentication, and analysing dataset
configurations.

## Prerequisites

### Environment

All scripts must run inside the project virtual-env:

```bash
source /home/Research/.venvs/siwn-tf216/bin/activate
```

Key packages: TensorFlow 2.16, Keras 3, scikit-learn, matplotlib, datasets,
pyarrow, huggingface_hub.

### Dataset

The default Hugging Face dataset is:

```
CAAI-FAU/PHY-Device-Fingerprinting-cuPHY-PUSCH
```

It ships several configurations (batches/slots/mobility). The four
pre-configured ones are listed in `train_hf_80_20.py` under
`DEFAULT_HF_CONFIGS`.

---

## Quick Start

```bash
cd /home/Research/AODT_RFFI
source /home/Research/.venvs/siwn-tf216/bin/activate

# 1 - Analyse available dataset configs (no GPU needed)
python fingerprinting/analyze_hf_dataset_configs.py

# 2 - Train + closed-set/open-set evaluation
python fingerprinting/train_hf_80_20.py   # edit __main__ block first

# 3 - Open-set only (reuse saved model, no retraining)
python fingerprinting/train_hf_80_20.py   # set RUN_OPEN_SET_ONLY = True
```

---

## 1. Dataset Analysis (`analyze_hf_dataset_configs.py`)

Generates per-configuration plots and a summary CSV to help pick the best
`required_iq_len` threshold.

### What it produces

For each config in `HF_CONFIGS`:

| Plot panel | Description |
|---|---|
| Samples per UE | Bar chart of total samples per UE label |
| IQ length distribution | Histogram of `nSym * nSc` across all rows |
| Retention vs required_iq_len | % of samples kept as the threshold increases |
| Per-UE IQ length stats | Min / median / max IQ length per UE |

Plus a cross-config retention comparison and `dataset_summary.csv`.

### Running

```bash
python fingerprinting/analyze_hf_dataset_configs.py
```

### Customisation

Edit the constants at the top of the file:

```python
HF_REPO_ID = "CAAI-FAU/PHY-Device-Fingerprinting-cuPHY-PUSCH"
HF_CONFIGS = [...]           # configs to analyse
OUTPUT_DIR = ".../dataset_analysis"
MIN_SAMPLES_PER_UE = 20      # threshold for "best required_iq_len" suggestion
```

### Output directory

```
aodt_hf_models/dataset_analysis/
├── <config>_analysis.png          # per-config 4-panel plot
├── cross_config_retention.png     # overlay retention curves
└── dataset_summary.csv            # tabular summary
```

---

## 2. Training + Evaluation (`train_hf_80_20.py`)

The main script. It trains a metric-learning feature extractor (TripletNet) on
known UEs with an 80/20 train/test split, evaluates closed-set accuracy, and
optionally runs enrollment-based open-set authentication.

### Configuration

All parameters are defined as plain variables inside the `if __name__ == "__main__":`
block. Edit them directly — no CLI flags needed for local runs.

#### Run mode flags

```python
RUN_OPEN_SET_ONLY = False     # True = skip training, load saved model
RUN_TRAINING_AND_TEST = True  # True = full pipeline
```

Set **exactly one** to `True`.

#### Key parameters

| Variable | Default | Description |
|---|---|---|
| `hf_config_name` | `DEFAULT_HF_CONFIGS[3]` | Which HF dataset configuration to use |
| `required_iq_len` | `39168` | Keep only records with this IQ length (`nSym*nSc`). Lower values retain more UEs but with shorter sequences. |
| `samples_count` | `400` | Number of IQ samples fed to the model (truncates to first N) |
| `backbone` | `"rnn"` | Feature extractor architecture: `"rnn"` (GRU-based) or `"cnn"` |
| `num_known_nodes` | `7` | Top-N most populated UEs used for training/closed-set testing |
| `num_open_set_nodes` | `3` | Least populated UEs reserved for open-set evaluation |
| `open_set_enroll_k` | `[10]` | List of K values: enrollment probes per open-set UE |
| `knn_k` | `5` | Number of neighbours for KNN classification |
| `train_ratio` | `0.8` | Fraction of known-UE data used for training (rest = closed-test) |

### Example: Train with RNN backbone, 7 known + 3 open-set UEs

```python
# In the __main__ block of train_hf_80_20.py:

RUN_OPEN_SET_ONLY = False
RUN_TRAINING_AND_TEST = True

hf_config_name = DEFAULT_HF_CONFIGS[3]  # 200batch-30slots-NoMobility
required_iq_len = 39168
backbone = "rnn"
num_known_nodes = 7
num_open_set_nodes = 3
open_set_enroll_k = [1, 3, 5, 10]
knn_k = 5
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python fingerprinting/train_hf_80_20.py
```

### Example: Train with CNN backbone on all available UEs

```python
RUN_OPEN_SET_ONLY = False
RUN_TRAINING_AND_TEST = True

backbone = "cnn"
num_known_nodes = None        # use all UEs
num_open_set_nodes = 0        # no open-set split
open_set_enroll_k = []
```

### What it produces

```
aodt_hf_models/
├── extractor_node1-1.keras              # saved feature extractor weights
└── plots/
    ├── label_distribution_node1-1.png   # train/closed-test/open-test bar chart
    ├── confusion_matrix_node1-1.png     # closed-set confusion (counts + normalised)
    ├── open_set_confusion_K10_node1-1.png   # open-set confusion per K
    └── open_set_per_ue_acc_K10_node1-1.png  # open-set per-UE accuracy bars
```

### Console output

```
=== AODT HF Training Configuration ===
...
Known split sizes: train=2211, closed_test=555 (ratio=0.7993)
Open-set test size: 197
...
Closed-set test accuracy (k=5): 0.9838

[Open-set] Enrollment-based authentication (K=10)...
  Open-set enrollment: K=10 probes/UE, qualifying UEs=[0, 1]
  Enrollment gallery: 20 probes (open-set only)
  Open-set test probes: 176
  Open-set authentication accuracy (K=10, knn_k=5): 0.8125
    UE 0: 55/61 correct (0.9016)
    UE 1: 88/115 correct (0.7652)
```

---

## 3. Open-Set Evaluation Only (`run_open_set_only`)

Reuses a previously trained model to test open-set authentication without
retraining. Useful for sweeping different K values or testing new open-set
splits.

### How it works

1. Loads the dataset with the same config used during training.
2. Reconstructs the model architecture and loads saved weights from
   `aodt_hf_models/extractor_<rx_id>.keras`.
3. Splits data into known / open-set using the same label selection logic.
4. For each open-set UE with more than K probes:
   - Randomly selects K probes as the **enrollment gallery**.
   - Embeds all probes through the frozen feature extractor.
   - Trains a KNN **only on the K enrollment embeddings** per UE.
   - Tests the remaining probes against the enrollment gallery.
5. Reports per-UE accuracy and saves confusion matrix / accuracy plots.

### Example: Sweep K = 1, 3, 5, 10, 20

```python
RUN_OPEN_SET_ONLY = True
RUN_TRAINING_AND_TEST = False

open_set_enroll_k = [1, 3, 5, 10, 20]
knn_k = 5
num_known_nodes = 7
num_open_set_nodes = 3
```

```bash
CUDA_VISIBLE_DEVICES=0 python fingerprinting/train_hf_80_20.py
```

### Example: Test with a different dataset config (same saved model)

```python
RUN_OPEN_SET_ONLY = True
RUN_TRAINING_AND_TEST = False

hf_config_name = DEFAULT_HF_CONFIGS[0]  # 500batch-10slots-0.01ms
open_set_enroll_k = [10]
```

---

## 4. Using CLI Arguments

The script also supports a full `argparse` interface via `parse_args()`.
To use it, replace the `__main__` block dispatch with:

```python
if __name__ == "__main__":
    args = parse_args()
    run_training_and_test(args)
```

Then run from the command line:

```bash
python fingerprinting/train_hf_80_20.py \
    --backbone rnn \
    --hf-config-name "data-10UE-1gNB-200batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB" \
    --required-iq-len 39168 \
    --num-known-nodes 7 \
    --num-open-set-nodes 3 \
    --open-set-enroll-k 1 3 5 10 \
    --knn-k 5
```

Run `python fingerprinting/train_hf_80_20.py --help` for the full list.

---

## Architecture Overview

### Pipeline

```
HuggingFace Dataset
    │
    ▼
 dataset_api.py ── load_hf_dataset() ── streaming + IQ length filter
    │
    ▼
 train_hf_80_20.py ── label selection (known vs open-set)
    │                   ── 80/20 per-device split
    │
    ├── dataset_preparation.py ── ChannelIndSpectrogram (STFT transform)
    │
    ├── deep_learning_models.py
    │       ├── TripletNet      (CNN backbone)
    │       ├── RNNTripletNet   (GRU backbone)
    │       └── QuadrupletNet   (CNN backbone)
    │
    ├── extractor_api.py ── train() / run() / load_feature_extractor()
    │
    ▼
 Evaluation
    ├── Closed-set:  KNN on train embeddings → predict test embeddings
    └── Open-set:    K enrollment probes → KNN gallery → predict remaining
```

### Key classes

| File | Class / Function | Purpose |
|---|---|---|
| `dataset_api.py` | `DatasetAPI` | Load HF datasets, filter, split |
| `dataset_preparation.py` | `ChannelIndSpectrogram` | IQ → spectrogram transform |
| `deep_learning_models.py` | `TripletNet` | CNN-based triplet feature extractor |
| `deep_learning_models.py` | `RNNTripletNet` | GRU-based triplet feature extractor |
| `extractor_api.py` | `ExtractorAPI` | Train / inference / model loading |
| `train_hf_80_20.py` | `run_training_and_test()` | Full train + eval pipeline |
| `train_hf_80_20.py` | `run_open_set_only()` | Eval-only with saved model |
| `analyze_hf_dataset_configs.py` | `main()` | Dataset analysis & plots |

### RNN backbone hyperparameters

When `backbone = "rnn"`, these control the GRU encoder:

| Variable | Default | Description |
|---|---|---|
| `rnn_gru_units` | `256` | Hidden units per GRU layer |
| `rnn_num_layers` | `2` | Number of stacked GRU layers |
| `rnn_dropout` | `0.3` | Dropout after each GRU |
| `rnn_recurrent_dropout` | `0.0` | Intra-recurrence dropout |
| `rnn_bidirectional` | `True` | Use bidirectional GRU |
| `rnn_embedding_dim` | `512` | Final embedding vector size |

---

## Troubleshooting

### Too few UEs after filtering

If `num_known_nodes = 7` but you only see 4 labels, `required_iq_len` is
filtering out UEs that don't have enough samples at that length. Run
`analyze_hf_dataset_configs.py` to find a better threshold, or lower
`required_iq_len`.

### CUDA / XLA errors

The script sets `TF_XLA_FLAGS` and disables MLIR graph optimisation
automatically. If you still see PTX errors, ensure `CUDA_VISIBLE_DEVICES`
points to a compatible GPU.

### Lambda layer deserialization error on model load

`run_open_set_only` avoids this by reconstructing the model architecture and
loading only the weights, instead of using `keras.models.load_model` directly.
Ensure `backbone` and all RNN hyperparameters in the `__main__` block match
those used during training.

### Dataset loading is slow

HF streaming can take 5-10 minutes for large configs. Progress is logged every
500 rows. If it stalls, check network connectivity to `huggingface.co`.
