"""
Train and evaluate RFFI on real srsRAN data from POWDER testbed.

Dataset: CAAI-FAU/powder-srsran-indoor (HuggingFace)
  - 3 configs: mar23, mar24_1, mar24_2
  - Each has 4 NUCs (nuc1-nuc4) with DMRS .mat files
  - DMRS shape per NUC: [N_slots, 1818] complex128

Usage:
  python fingerprinting/train_powder_srsran.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_DISABLE_MLIR_GRAPH_OPTIMIZATION", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from extractor_api import ExtractorAPI
from dataset_preparation import ChannelIndSpectrogram

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO = "CAAI-FAU/powder-srsran-indoor"
TRAIN_CONFIGS = ["powder_debug_mar24_1", "powder_debug_mar24_2"]
TEST_CONFIGS = ["powder_debug_mar23"]
NUC_IDS = ["nuc1", "nuc2", "nuc3", "nuc4"]

# How many IQ samples (complex) to use per slot from the 1818 DMRS values
SAMPLES_COUNT = 400

# Model hyperparameters
BACKBONE = "rnn"           # "cnn" or "rnn"
LOSS_TYPE = "triplet_loss"
ALPHA = 1.1
BATCH_SIZE = 256
ROW = 80                   # STFT window size (must be 80 for guard removal)
ENABLE_IND = True          # Channel-independent spectrogram

# RNN hyperparameters
RNN_GRU_UNITS = 256
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.3
RNN_RECURRENT_DROPOUT = 0.0
RNN_BIDIRECTIONAL = True
RNN_EMBEDDING_DIM = 512

# Evaluation
KNN_K = 5
KNN_PROBES_PER_DEVICE = 100  # Max probes per device for KNN fitting
PREDICT_BATCH_SIZE = 2048    # Batch size for model.predict() to avoid OpenBLAS OOM

# Output
RX_ID = "powder-srsran"
MODEL_DIR = None           # defaults to <repo_root>/aodt_hf_models
PLOT_DIR = None            # defaults to <model_dir>/plots

TEST_ONLY = True

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dmrs_mat(repo_id, config_prefix, nuc_id):
    """Download and load a single dmrs.mat file from HuggingFace."""
    from huggingface_hub import hf_hub_download
    import scipy.io as sio

    mat_path = f"{config_prefix}/{nuc_id}/dmrs.mat"
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=mat_path)
    mat = sio.loadmat(local_path)

    # The key is 'dmrsMatrix' — complex128, shape [N_slots, 1818]
    dmrs = mat["dmrsMatrix"].astype(np.complex64)
    
    return dmrs


def load_powder_dataset(repo_id, configs, nuc_ids, samples_count):
    """
    Load DMRS data from multiple configs, assign integer labels from NUC IDs.

    Returns:
        data: np.ndarray [N_total, samples_count] complex64
        labels: np.ndarray [N_total] int
        label_names: dict mapping int label -> nuc string
    """
    label_map = {nuc: idx for idx, nuc in enumerate(nuc_ids)}
    label_names = {idx: nuc for nuc, idx in label_map.items()}

    all_data = []
    all_labels = []

    for config in configs:
        for nuc in nuc_ids:
            print(f"  Loading {config}/{nuc}...", end=" ")
            dmrs = load_dmrs_mat(repo_id, config, nuc)
            n_slots = dmrs.shape[0]
            # Truncate each slot's DMRS to samples_count
            dmrs_trunc = dmrs[:, :samples_count]
            all_data.append(dmrs_trunc)
            all_labels.append(np.full(n_slots, label_map[nuc], dtype=int))
            print(f"{n_slots} slots")
            # Get the average IQ per slot
            print(f" Number of samples ")

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Shuffle
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(labels))
    data = data[perm]
    labels = labels[perm]

    return data, labels, label_names


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion(labels_true, labels_pred, labels_order, label_names, output_path, title_suffix=""):
    """Plot side-by-side count and normalized confusion matrices."""
    cm = confusion_matrix(labels_true, labels_pred, labels=labels_order)
    cm = cm.astype(np.int32)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float32), where=row_sums > 0)

    tick_labels = [label_names.get(x, str(x)) for x in labels_order]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=140)

    # Counts
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title(f"Confusion Matrix (Counts){title_suffix}")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].set_xticks(np.arange(len(labels_order)))
    axes[0].set_yticks(np.arange(len(labels_order)))
    axes[0].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[0].set_yticklabels(tick_labels)
    max_count = max(1, int(cm.max()))
    count_thresh = max_count / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            text_color = "white" if val > count_thresh else "black"
            axes[0].text(j, i, f"{val}", ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Row-normalized
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Oranges", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Confusion Matrix (Row-normalized){title_suffix}")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    axes[1].set_xticks(np.arange(len(labels_order)))
    axes[1].set_yticks(np.arange(len(labels_order)))
    axes[1].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[1].set_yticklabels(tick_labels)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            pct = float(cm_norm[i, j]) * 100.0
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            axes[1].text(j, i, f"{pct:.1f}%", ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return cm


def plot_label_distribution(labels_train, labels_test, labels_order, label_names, output_path):
    """Bar chart of per-device sample counts for train and test sets."""
    train_counts = [int(np.sum(labels_train == lbl)) for lbl in labels_order]
    test_counts = [int(np.sum(labels_test == lbl)) for lbl in labels_order]
    tick_labels = [label_names.get(x, str(x)) for x in labels_order]

    x = np.arange(len(labels_order))
    width = 0.35

    plt.figure(figsize=(10, 5), dpi=140)
    bars_train = plt.bar(x - width / 2, train_counts, width=width, label="Train (mar24_1+2)", color="#1f77b4")
    bars_test = plt.bar(x + width / 2, test_counts, width=width, label="Test (mar23)", color="#ff7f0e")
    plt.xticks(x, tick_labels)
    plt.xlabel("Device ID")
    plt.ylabel("Number of slots")
    plt.title("Sample Distribution per Device (Train / Test)")
    plt.legend()

    max_height = max(train_counts + test_counts) if (train_counts or test_counts) else 1
    y_offset = max(1, int(max_height * 0.01))
    for bar in list(bars_train) + list(bars_test):
        height = int(bar.get_height())
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + y_offset, str(height),
                 ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference_batched(model, data, model_config, batch_size):
    """Run feature extraction in batches to avoid OpenBLAS thread/memory limits."""
    cis = ChannelIndSpectrogram()
    n = data.shape[0]
    fps_parts = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = data[start:end]
        spec = cis.channel_ind_spectrogram(chunk, model_config['row'], enable_ind=model_config['enable_ind'])
        fps = model.predict(spec, verbose=0)
        fps_parts.append(fps)
        print(f"  {end}/{n}", end="\r")
    print()
    return np.concatenate(fps_parts, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root_dir = str(Path(__file__).resolve().parents[1])
    model_dir = MODEL_DIR or os.path.join(root_dir, "aodt_hf_models")
    plot_dir = PLOT_DIR or os.path.join(model_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    model_config = {
        "batch_size": BATCH_SIZE,
        "loss_type": LOSS_TYPE,
        "backbone": BACKBONE,
        "alpha": ALPHA,
        "row": ROW,
        "enable_ind": ENABLE_IND,
        "rnn_gru_units": RNN_GRU_UNITS,
        "rnn_num_layers": RNN_NUM_LAYERS,
        "rnn_dropout": RNN_DROPOUT,
        "rnn_recurrent_dropout": RNN_RECURRENT_DROPOUT,
        "rnn_bidirectional": RNN_BIDIRECTIONAL,
        "rnn_embedding_dim": RNN_EMBEDDING_DIM,
    }

    extractor_api = ExtractorAPI()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=== Powder-srsRAN RFFI Training ===")
    print(f"Python: {sys.executable}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Train configs: {TRAIN_CONFIGS}")
    print(f"Test configs:  {TEST_CONFIGS}")
    print(f"NUCs: {NUC_IDS}")
    print(f"Samples/slot: {SAMPLES_COUNT}")
    print(f"Backbone: {BACKBONE}")
    
    print("Loading training data (mar24_1 + mar24_2)...")
    data_train, labels_train, label_names = load_powder_dataset(
        HF_REPO, TRAIN_CONFIGS, NUC_IDS, SAMPLES_COUNT,
    )
    print(f"  Train total: {data_train.shape[0]} slots, shape: {data_train.shape}")
    print("Shape of training data for loading model: ", data_train.shape)

    print("\n Loading test data (mar23)...")
    data_test, labels_test, _ = load_powder_dataset(
        HF_REPO, TEST_CONFIGS, NUC_IDS, SAMPLES_COUNT,
    )
    print(f"  Test total: {data_test.shape[0]} slots, shape: {data_test.shape}")

    labels_order = sorted(set(labels_train) | set(labels_test))
    print(f"\nLabel mapping: {label_names}")
    for lbl in labels_order:
        n_train = int(np.sum(labels_train == lbl))
        n_test = int(np.sum(labels_test == lbl))
        print(f"  {label_names[lbl]} (label={lbl}): train={n_train}, test={n_test}")

    # Plot distribution
    plot_label_distribution(
        labels_train, labels_test, labels_order, label_names,
        os.path.join(plot_dir, f"label_distribution_{RX_ID}.png"),
    )
    model_file = os.path.join(model_dir, f"extractor_{RX_ID}.keras")
    
    if not TEST_ONLY:
        # ------------------------------------------------------------------
        # 2. Train
        # ------------------------------------------------------------------
        print(f"\n Training {BACKBONE.upper()} feature extractor...")
        
        feature_extractor, history_obj = extractor_api.train(
            data_train, labels_train, labels_order, model_config, save_path=model_file,
        )
        print(f"Training complete. Saved: {model_file}")
        history = history_obj.history
        if "loss" in history and len(history["loss"]) > 0:
            print(f"  Final train loss: {history['loss'][-1]:.6f}")
        if "val_loss" in history and len(history["val_loss"]) > 0:
            print(f"  Final valid loss: {history['val_loss'][-1]:.6f}")
    else:
        print("Shape of training data for loading model: ", data_train.shape)
        # data_train_shape = Channel_ind_spectrogram(data_train, row=ROW, enable_ind=ENABLE_IND)
        # Create single channel independent spectrogram for loading data shape
        data_train_shape = ChannelIndSpectrogram().channel_ind_spectrogram(data_train[0:1], row=ROW, enable_ind=ENABLE_IND)
        feature_extractor = extractor_api.load_feature_extractor(
            model_file, 
            model_config, 
            datashape=data_train_shape.shape
        )

    # ------------------------------------------------------------------
    # 3. Evaluate — closed-set KNN
    # ------------------------------------------------------------------

    # Subsample training data for KNN fitting (avoids OpenBLAS OOM)
    rng = np.random.RandomState(42)
    knn_idx = []
    for lbl in labels_order:
        lbl_idx = np.where(labels_train == lbl)[0]
        if len(lbl_idx) > KNN_PROBES_PER_DEVICE:
            lbl_idx = rng.choice(lbl_idx, KNN_PROBES_PER_DEVICE, replace=False)
        knn_idx.extend(lbl_idx)
    knn_idx = np.array(knn_idx)
    data_train_knn = data_train[knn_idx]
    labels_train_knn = labels_train[knn_idx]
    print(f"\nKNN gallery: {len(knn_idx)} probes ({KNN_PROBES_PER_DEVICE}/device)")

    print("Extracting fingerprints (train gallery)...")
    fps_train = _run_inference_batched(feature_extractor, data_train_knn, model_config, PREDICT_BATCH_SIZE)
    print(f"Extracting fingerprints (test, {data_test.shape[0]} samples)...")
    fps_test = _run_inference_batched(feature_extractor, data_test, model_config, PREDICT_BATCH_SIZE)

    knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean")
    knn.fit(fps_train, labels_train_knn)
    labels_pred = knn.predict(fps_test)
    accuracy = accuracy_score(labels_test, labels_pred)
    print(f"\nClosed-set accuracy (KNN k={KNN_K}): {accuracy:.4f}")

    # Per-device accuracy
    for lbl in labels_order:
        mask = labels_test == lbl
        if mask.sum() == 0:
            continue
        acc = accuracy_score(labels_test[mask], labels_pred[mask])
        print(f"  {label_names[lbl]}: {acc:.4f} ({int(mask.sum())} samples)")

    # Confusion matrix
    cm_path = os.path.join(plot_dir, f"confusion_matrix_{RX_ID}.png")
    cm = plot_confusion(
        labels_test, labels_pred, labels_order, label_names, cm_path,
        title_suffix=f"\nTrain: mar24_1+mar24_2, Test: mar23 (acc={accuracy:.2%})",
    )

    # Print top confusions
    failures = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                failures.append((int(cm[i, j]), labels_order[i], labels_order[j]))
    failures.sort(reverse=True, key=lambda x: x[0])
    if failures:
        print("\nTop confusion pairs (true -> predicted):")
        for cnt, y_true, y_pred in failures[:5]:
            print(f"  {label_names[y_true]} -> {label_names[y_pred]}: {cnt}")

    print("\nDone.")


if __name__ == "__main__":
    main()
