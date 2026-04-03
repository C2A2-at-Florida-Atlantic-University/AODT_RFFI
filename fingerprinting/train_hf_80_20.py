import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

# Reduce GPU/XLA kernel-gen incompatibility risk on some container/driver combos.
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_DISABLE_MLIR_GRAPH_OPTIMIZATION", "1")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from dataset_api import DatasetAPI
from extractor_api import ExtractorAPI


DEFAULT_HF_REPO = "CAAI-FAU/PHY-Device-Fingerprinting-cuPHY-PUSCH"
DEFAULT_HF_CONFIGS = [
    "data-10UE-1gNB-500batch-10slots-1sample-0.01ms-halfwaveDipole_UE_gNB",
    "data-10UE-1gNB-500batch-10slots-1sample-NoMobility-halfwaveDipole_UE_gNB",
    "data-10UE-1gNB-100batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB",
    "data-10UE-1gNB-200batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB", # EBC
    "data-10UE-1gNB-200batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB-Hospital",
    "data-10UE-1gNB-200batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB-USTAR",
    "data-10UE-1gNB-200batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB-GuestHouse",
    ]
DEFAULT_HF_CONFIG = DEFAULT_HF_CONFIGS[6]

def build_data_config(args, model_path):
    return {
        "dataset_name": DatasetAPI.DATASET_AODT_HF,
        "samples_count": args.samples_count,
        "hf_required_iq_len": args.required_iq_len,
        "hf_repo_id": args.hf_repo_id,
        "hf_config_name": args.hf_config_name,
        "hf_revision": args.hf_revision,
        "hf_train_split": args.hf_train_split,
        "hf_test_split": args.hf_test_split,
        "hf_train_ratio": args.train_ratio,
        "hf_label_column": args.label_column,
        "hf_iq_column": args.iq_column,
        "hf_rx_ant": args.rx_ant,
        "hf_sym_mode": args.sym_mode,
        "hf_max_train_samples": args.max_train_samples,
        "hf_max_test_samples": args.max_test_samples,
        "model_path": model_path,
    }


def build_model_config(args):
    model_config = {
        "batch_size": args.batch_size,
        "loss_type": args.loss_type,
        "backbone": args.backbone,
        "alpha": args.alpha,
        "row": args.row,
        "enable_ind": args.enable_ind,
    }
    if args.backbone == "rnn":
        model_config["rnn_gru_units"] = args.rnn_gru_units
        model_config["rnn_num_layers"] = args.rnn_num_layers
        model_config["rnn_dropout"] = args.rnn_dropout
        model_config["rnn_recurrent_dropout"] = args.rnn_recurrent_dropout
        model_config["rnn_bidirectional"] = args.rnn_bidirectional
        model_config["rnn_embedding_dim"] = args.rnn_embedding_dim
    if args.loss_type == "quadruplet_loss":
        model_config["beta"] = args.beta
    return model_config


def _plot_label_distribution(labels_train, labels_closed_test, labels_open_test, labels_order, output_path):
    train_counts = [int(np.sum(labels_train == lbl)) for lbl in labels_order]
    closed_counts = [int(np.sum(labels_closed_test == lbl)) for lbl in labels_order]
    open_counts = [int(np.sum(labels_open_test == lbl)) for lbl in labels_order]

    x = np.arange(len(labels_order))
    width = 0.26

    plt.figure(figsize=(10, 5), dpi=140)
    bars_train = plt.bar(x - width, train_counts, width=width, label="Train", color="#1f77b4")
    bars_closed = plt.bar(x, closed_counts, width=width, label="Closed-test", color="#ff7f0e")
    bars_open = plt.bar(x + width, open_counts, width=width, label="Open-test", color="#2ca02c")
    plt.xticks(x, [str(lbl) for lbl in labels_order])
    plt.xlabel("Device ID (label)")
    plt.ylabel("Number of samples")
    plt.title("Sample Distribution per Device ID (Train / Closed-test / Open-test)")
    plt.legend()
    max_height = max(train_counts + closed_counts + open_counts) if (train_counts or closed_counts or open_counts) else 1
    y_offset = max(1, int(max_height * 0.01))
    for bar in list(bars_train) + list(bars_closed) + list(bars_open):
        height = int(bar.get_height())
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset,
            str(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _plot_confusion(labels_true, labels_pred, labels_order, output_path):
    cm = confusion_matrix(labels_true, labels_pred, labels=labels_order)
    cm = cm.astype(np.int32)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float32), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=140)

    # Absolute confusion counts
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].set_xticks(np.arange(len(labels_order)))
    axes[0].set_yticks(np.arange(len(labels_order)))
    axes[0].set_xticklabels([str(x) for x in labels_order], rotation=45, ha="right")
    axes[0].set_yticklabels([str(x) for x in labels_order])
    max_count = max(1, int(cm.max()))
    count_thresh = max_count / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            text_color = "white" if val > count_thresh else "black"
            axes[0].text(j, i, f"{val}", ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Normalized confusion (row-wise)
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Oranges", vmin=0.0, vmax=1.0)
    axes[1].set_title("Confusion Matrix (Row-normalized)")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    axes[1].set_xticks(np.arange(len(labels_order)))
    axes[1].set_yticks(np.arange(len(labels_order)))
    axes[1].set_xticklabels([str(x) for x in labels_order], rotation=45, ha="right")
    axes[1].set_yticklabels([str(x) for x in labels_order])
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            pct = float(cm_norm[i, j]) * 100.0
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            axes[1].text(j, i, f"{pct:.1f}%", ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return cm


def _plot_open_set_results(
    labels_true,
    labels_pred,
    labels_order,
    per_ue_acc,
    enroll_k,
    output_path_cm,
    output_path_bar,
):
    cm = confusion_matrix(labels_true, labels_pred, labels=labels_order)
    cm = cm.astype(np.int32)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float32), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=140)

    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title(f"Open-set Confusion (Counts, enroll K={enroll_k})")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].set_xticks(np.arange(len(labels_order)))
    axes[0].set_yticks(np.arange(len(labels_order)))
    axes[0].set_xticklabels([str(x) for x in labels_order], rotation=45, ha="right")
    axes[0].set_yticklabels([str(x) for x in labels_order])
    max_count = max(1, int(cm.max()))
    count_thresh = max_count / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            text_color = "white" if val > count_thresh else "black"
            axes[0].text(j, i, f"{val}", ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Oranges", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Open-set Confusion (Row-normalized, enroll K={enroll_k})")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    axes[1].set_xticks(np.arange(len(labels_order)))
    axes[1].set_yticks(np.arange(len(labels_order)))
    axes[1].set_xticklabels([str(x) for x in labels_order], rotation=45, ha="right")
    axes[1].set_yticklabels([str(x) for x in labels_order])
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            pct = float(cm_norm[i, j]) * 100.0
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            axes[1].text(j, i, f"{pct:.1f}%", ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path_cm, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved open-set confusion plot: {output_path_cm}")

    ue_labels_sorted = sorted(per_ue_acc.keys())
    accs = [per_ue_acc[u] for u in ue_labels_sorted]
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=140)
    bars = ax2.bar([str(u) for u in ue_labels_sorted], accs, color="#9467bd")
    ax2.set_xlabel("Open-set UE label")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Per-UE Open-set Auth Accuracy (enroll K={enroll_k})")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8)
    for bar, acc_val in zip(bars, accs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            acc_val + 0.01,
            f"{acc_val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig2.tight_layout()
    fig2.savefig(output_path_bar, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved open-set per-UE accuracy plot: {output_path_bar}")

    return cm


def _run_open_set_evaluation(
    feature_extractor,
    extractor_api,
    model_config,
    data_open,
    labels_open,
    enroll_k,
    knn_k,
    plot_dir,
    rx_id,
    plot_outputs=True,
    seed=42,
):
    rng = np.random.RandomState(seed)
    open_ue_labels = sorted(np.unique(labels_open).tolist())

    enroll_data_parts = []
    enroll_label_parts = []
    test_data_parts = []
    test_label_parts = []
    skipped = []

    for ue in open_ue_labels:
        mask = labels_open == ue
        ue_data = data_open[mask]
        ue_count = ue_data.shape[0]
        if ue_count <= enroll_k:
            skipped.append((ue, ue_count))
            continue
        idxs = rng.permutation(ue_count)
        enroll_idx = idxs[:enroll_k]
        test_idx = idxs[enroll_k:]
        enroll_data_parts.append(ue_data[enroll_idx])
        enroll_label_parts.append(np.full(enroll_k, ue, dtype=int))
        test_data_parts.append(ue_data[test_idx])
        test_label_parts.append(np.full(len(test_idx), ue, dtype=int))

    if skipped:
        for ue, cnt in skipped:
            print(f"  [WARN] Open-set UE {ue} has only {cnt} probes (<= K={enroll_k}), skipped from open-set eval.")

    if not test_data_parts:
        print("  [WARN] No open-set UEs qualify for enrollment-based testing (all have <= K probes). Skipping.")
        return

    enroll_data = np.concatenate(enroll_data_parts, axis=0)
    enroll_labels = np.concatenate(enroll_label_parts, axis=0)
    test_data = np.concatenate(test_data_parts, axis=0)
    test_labels = np.concatenate(test_label_parts, axis=0)
    qualifying_ues = sorted(np.unique(enroll_labels).tolist())

    print(f"  Open-set enrollment: K={enroll_k} probes/UE, qualifying UEs={qualifying_ues}")
    print(f"  Enrollment gallery: {enroll_data.shape[0]} probes (open-set only)")
    print(f"  Open-set test probes: {test_data.shape[0]}")

    fps_enroll = extractor_api.run(feature_extractor, enroll_data, model_config)
    fps_test = extractor_api.run(feature_extractor, test_data, model_config)

    gallery_fps = fps_enroll
    gallery_labels = enroll_labels

    effective_k = min(knn_k, gallery_fps.shape[0])
    knn = KNeighborsClassifier(n_neighbors=effective_k, metric="euclidean")
    knn.fit(gallery_fps, gallery_labels)

    test_pred = knn.predict(fps_test)
    overall_acc = accuracy_score(test_labels, test_pred)
    print(f"  Open-set authentication accuracy (K={enroll_k}, knn_k={effective_k}): {overall_acc:.4f}")

    per_ue_acc = {}
    for ue in qualifying_ues:
        ue_mask = test_labels == ue
        if ue_mask.sum() == 0:
            continue
        ue_acc = accuracy_score(test_labels[ue_mask], test_pred[ue_mask])
        ue_test_n = int(ue_mask.sum())
        ue_correct = int(np.sum(test_pred[ue_mask] == ue))
        per_ue_acc[ue] = ue_acc
        print(f"    UE {ue}: {ue_correct}/{ue_test_n} correct ({ue_acc:.4f})")

    if plot_outputs and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        cm_path = os.path.join(plot_dir, f"open_set_confusion_K{enroll_k}_{rx_id}.png")
        bar_path = os.path.join(plot_dir, f"open_set_per_ue_acc_K{enroll_k}_{rx_id}.png")
        _plot_open_set_results(
            test_labels, test_pred, qualifying_ues,
            per_ue_acc, enroll_k, cm_path, bar_path,
        )

    return {
        "enroll_k": enroll_k,
        "qualifying_ues": qualifying_ues,
        "overall_accuracy": overall_acc,
        "per_ue_accuracy": per_ue_acc,
        "total_test_probes": test_data.shape[0],
    }


def _print_top_confusions(cm, labels_order, top_k=5):
    failures = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            cnt = int(cm[i, j])
            if cnt > 0:
                failures.append((cnt, labels_order[i], labels_order[j]))

    failures.sort(reverse=True, key=lambda x: x[0])
    if not failures:
        print("No off-diagonal confusions found.")
        return

    print("Top confusion pairs (true -> predicted):")
    for cnt, y_true, y_pred in failures[:top_k]:
        print(f"  {y_true} -> {y_pred}: {cnt}")


def _prepare_dataset(args):
    """Load HF dataset, select known/open-set labels, split, and return all artifacts."""
    root_dir = str(Path(__file__).resolve().parents[1])
    model_path = args.model_path or os.path.join(root_dir, "aodt_hf_models")
    os.makedirs(model_path, exist_ok=True)

    data_config = build_data_config(args, model_path=model_path)
    model_config = build_model_config(args)

    dataset_api = DatasetAPI(
        root_dir=root_dir, matlab_src_dir="", matlab_session_id="", aug_on=False,
    )
    extractor_api = ExtractorAPI()

    data_all, labels_all, _ = dataset_api.load_hf_dataset(
        repo_id=data_config["hf_repo_id"],
        split=data_config["hf_train_split"],
        revision=data_config.get("hf_revision"),
        config_name=data_config.get("hf_config_name"),
        label_column=data_config["hf_label_column"],
        iq_column=data_config["hf_iq_column"],
        rx_ant=data_config["hf_rx_ant"],
        sym_mode=data_config["hf_sym_mode"],
        batch_filter=data_config.get("hf_train_batches"),
        slot_filter=data_config.get("hf_train_slots"),
        max_samples=data_config.get("hf_max_train_samples"),
        shuffle=False,
        required_iq_len=data_config.get("hf_required_iq_len"),
        prefer_parquet_loader=False,
    )
    data_all = data_all[:, 0 : data_config["samples_count"]]
    labels_all = labels_all.flatten().astype(int)

    unique_labels, label_counts = np.unique(labels_all, return_counts=True)
    label_count_pairs = sorted(zip(unique_labels.tolist(), label_counts.tolist()), key=lambda x: (x[1], x[0]))
    label_count_pairs_desc = sorted(label_count_pairs, key=lambda x: (-x[1], x[0]))

    if args.num_known_nodes is not None:
        if args.num_known_nodes < 2:
            raise ValueError("--num-known-nodes must be >= 2 for triplet training.")
        known_labels = sorted([lbl for lbl, _ in label_count_pairs_desc[: args.num_known_nodes]])
        if len(known_labels) < args.num_known_nodes:
            print(
                "[WARN] Requested num_known_nodes="
                f"{args.num_known_nodes}, but only {len(known_labels)} labels remain after loading/filtering."
            )
            print(
                "[WARN] This usually means hf_required_iq_len and/or dataset config removed other labels. "
                "Try a lower required_iq_len or a different HF config."
            )
        remaining_for_open = [lbl for lbl, _ in label_count_pairs if lbl not in known_labels]
        open_set_labels = remaining_for_open[: args.num_open_set_nodes]
    else:
        open_set_labels = [lbl for lbl, _ in label_count_pairs[: args.num_open_set_nodes]]
        known_labels = [lbl for lbl, _ in label_count_pairs if lbl not in open_set_labels]

    if not known_labels:
        raise RuntimeError("No known labels remain after selecting open-set labels.")

    if args.num_known_nodes is not None:
        print(f"Selected known labels for training/eval (top-{args.num_known_nodes} by count): {known_labels}")
    print(f"Selected open-set labels (least samples): {open_set_labels}")
    for lbl, cnt in label_count_pairs:
        tag = "open-set" if lbl in open_set_labels else "known"
        print(f"  label={lbl}: count={cnt} [{tag}]")

    known_mask = np.isin(labels_all, known_labels)
    open_mask = np.isin(labels_all, open_set_labels)

    data_known = data_all[known_mask]
    labels_known = labels_all[known_mask].reshape(-1, 1)
    data_open = data_all[open_mask]
    labels_open = labels_all[open_mask]

    (
        data_train, labels_train, _,
        data_closed_test, labels_closed_test, _,
    ) = dataset_api._split_by_device_ratio(data_known, labels_known, rssi=None, train_ratio=args.train_ratio)

    data_train, labels_train, _ = dataset_api._shuffle_dataset(data_train, labels_train, None)
    labels_train = labels_train.flatten().astype(int)
    labels_closed_test = labels_closed_test.flatten().astype(int)

    print(
        f"Known split sizes: train={data_train.shape[0]}, closed_test={data_closed_test.shape[0]} "
        f"(ratio={data_train.shape[0] / max(1, (data_train.shape[0] + data_closed_test.shape[0])):.4f})"
    )
    print(f"Open-set test size: {data_open.shape[0]}")

    return SimpleNamespace(
        data_train=data_train, labels_train=labels_train,
        data_closed_test=data_closed_test, labels_closed_test=labels_closed_test,
        data_open=data_open, labels_open=labels_open,
        known_labels=known_labels, open_set_labels=open_set_labels,
        label_count_pairs=label_count_pairs,
        dataset_api=dataset_api, extractor_api=extractor_api,
        data_config=data_config, model_config=model_config,
        model_path=model_path, data_all=data_all,
    )


def run_training_and_test(args):
    print("=== AODT HF Training Configuration ===")
    print(f"Python executable: {sys.executable}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"HF repo: {args.hf_repo_id}")
    print(f"HF config: {args.hf_config_name}")
    print(f"Split mode: train='{args.hf_train_split}', test='{args.hf_test_split}'")
    print(f"Requested train ratio: {args.train_ratio:.4f}")
    print(f"Required IQ length: {args.required_iq_len}")
    print(f"RX ID: {args.rx_id}")
    print(f"Samples count: {args.samples_count}")
    print()

    print("[1/2] Preparing dataset split and training feature extractor...")
    ds = _prepare_dataset(args)
    model_path = ds.model_path
    model_config = ds.model_config
    extractor_api = ds.extractor_api

    if args.plot_outputs:
        plot_dir = args.plot_dir or os.path.join(model_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        labels_order = sorted(list(set(ds.labels_train).union(set(ds.labels_closed_test)).union(set(ds.labels_open))))
        dist_plot = os.path.join(plot_dir, f"label_distribution_{args.rx_id}.png")
        _plot_label_distribution(ds.labels_train, ds.labels_closed_test, ds.labels_open, labels_order, dist_plot)

    model_file = os.path.join(model_path, f"extractor_{args.rx_id}.keras")
    feature_extractor, history_obj = extractor_api.train(
        ds.data_train, ds.labels_train, ds.known_labels, model_config, save_path=model_file,
    )
    print("Training complete.")
    print(f"Saved model: {model_file}")
    if args.print_history:
        history = history_obj.history
        if "loss" in history and len(history["loss"]) > 0:
            print(f"Final train loss: {history['loss'][-1]:.6f}")
        if "val_loss" in history and len(history["val_loss"]) > 0:
            print(f"Final valid loss: {history['val_loss'][-1]:.6f}")

    if not args.skip_eval:
        print("\n[2/2] Evaluating on held-out test split...")
        fps_train = extractor_api.run(feature_extractor, ds.data_train, model_config)
        fps_test = extractor_api.run(feature_extractor, ds.data_closed_test, model_config)
        knn = KNeighborsClassifier(n_neighbors=args.knn_k, metric="euclidean")
        knn.fit(fps_train, ds.labels_train)
        labels_pred = knn.predict(fps_test)
        accuracy = accuracy_score(ds.labels_closed_test, labels_pred)
        print(f"Closed-set test accuracy (k={args.knn_k}): {accuracy:.4f}")

        if args.plot_outputs:
            plot_dir = args.plot_dir or os.path.join(model_path, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            cm_plot = os.path.join(plot_dir, f"confusion_matrix_{args.rx_id}.png")

            cm_labels_order = sorted(list(set(ds.labels_train).union(set(ds.labels_closed_test))))
            cm = _plot_confusion(ds.labels_closed_test, labels_pred, cm_labels_order, cm_plot)
            _print_top_confusions(cm, cm_labels_order, top_k=args.top_confusions)

            print(f"Saved confusion matrix plot: {cm_plot}")

        if ds.data_open.shape[0] > 0:
            enroll_k_values = args.open_set_enroll_k if isinstance(args.open_set_enroll_k, (list, tuple)) else [args.open_set_enroll_k]
            for ek in enroll_k_values:
                print(f"\n[Open-set] Enrollment-based authentication (K={ek})...")
                _run_open_set_evaluation(
                    feature_extractor=feature_extractor,
                    extractor_api=extractor_api,
                    model_config=model_config,
                    data_open=ds.data_open,
                    labels_open=ds.labels_open,
                    enroll_k=ek,
                    knn_k=args.knn_k,
                    plot_dir=args.plot_dir or os.path.join(model_path, "plots"),
                    rx_id=args.rx_id,
                    plot_outputs=args.plot_outputs,
                )
        else:
            print("\nNo open-set UEs available; skipping enrollment-based open-set eval.")
    else:
        print("\nEvaluation skipped (--skip-eval enabled).")

    return feature_extractor


def run_open_set_only(args):
    """Load a pre-trained model and run open-set enrollment evaluation only."""
    from dataset_preparation import ChannelIndSpectrogram

    print("=== AODT HF Open-set Evaluation (no training) ===")
    print("Loading dataset...")
    ds = _prepare_dataset(args)

    model_file = os.path.join(ds.model_path, f"extractor_{args.rx_id}.keras")

    dummy_spec = ChannelIndSpectrogram().channel_ind_spectrogram(
        ds.data_all[:1], ds.model_config["row"], enable_ind=ds.model_config["enable_ind"],
    )
    datashape = (ds.data_all.shape[0],) + dummy_spec.shape[1:]

    print(f"Loading model: {model_file}")
    feature_extractor = ds.extractor_api.load_feature_extractor(model_file, ds.model_config, datashape)
    print(f"Model loaded. Output shape: {feature_extractor.output_shape}")

    if ds.data_open.shape[0] == 0:
        print("[WARN] No open-set data available. Nothing to evaluate.")
        return

    plot_dir = args.plot_dir or os.path.join(ds.model_path, "plots")
    enroll_k_values = args.open_set_enroll_k if isinstance(args.open_set_enroll_k, (list, tuple)) else [args.open_set_enroll_k]

    for ek in enroll_k_values:
        print(f"\n[Open-set] Enrollment-based authentication (K={ek})...")
        _run_open_set_evaluation(
            feature_extractor=feature_extractor,
            extractor_api=ds.extractor_api,
            model_config=ds.model_config,
            data_open=ds.data_open,
            labels_open=ds.labels_open,
            enroll_k=ek,
            knn_k=args.knn_k,
            plot_dir=plot_dir,
            rx_id=args.rx_id,
            plot_outputs=args.plot_outputs,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train HF fingerprint extractor on "
            "CAAI-FAU/PHY-Device-Fingerprinting-cuPHY-PUSCH using 80/20 split."
        )
    )
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO, help="HF dataset repo id")
    parser.add_argument(
        "--hf-config-name",
        default=DEFAULT_HF_CONFIG,
        help="HF dataset config name",
    )
    parser.add_argument("--hf-revision", default=None, help="HF dataset revision/tag/branch")
    parser.add_argument("--hf-train-split", default="train", help="HF train split")
    parser.add_argument(
        "--hf-test-split",
        default="train",
        help="HF test split; keep equal to train to use ratio split",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction (default: 0.8)")
    parser.add_argument("--label-column", default="rnti", help="Label column in dataset")
    parser.add_argument("--iq-column", default="iq", help="IQ column in dataset")
    parser.add_argument("--rx-ant", type=int, default=0, help="RX antenna index")
    parser.add_argument(
        "--sym-mode",
        default="flatten",
        choices=["flatten", "first_sym", "mean_sym"],
        help="How IQ symbols are collapsed to 1D",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on train load")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap on test load")
    parser.add_argument("--samples-count", type=int, default=400, help="Number of IQ samples used")
    parser.add_argument(
        "--required-iq-len",
        type=int,
        default=39168,
        help="Keep only records with this exact post-processed IQ length",
    )
    parser.add_argument("--model-path", default=None, help="Directory where models are saved")
    parser.add_argument("--rx-id", default=DatasetAPI.RX_1, help="Receiver/model id suffix")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--loss-type",
        default="triplet_loss",
        choices=["triplet_loss", "quadruplet_loss"],
        help="Metric learning loss type",
    )
    parser.add_argument(
        "--backbone",
        default="rnn",
        choices=["cnn", "rnn"],
        help="Encoder backbone for triplet training",
    )
    parser.add_argument("--alpha", type=float, default=1.1, help="Triplet/quadruplet alpha")
    parser.add_argument("--beta", type=float, default=0.37, help="Quadruplet beta")
    parser.add_argument("--rnn-gru-units", type=int, default=256, help="RNN backbone GRU units")
    parser.add_argument("--rnn-num-layers", type=int, default=2, help="RNN backbone number of GRU layers")
    parser.add_argument("--rnn-dropout", type=float, default=0.3, help="RNN backbone dropout")
    parser.add_argument("--rnn-recurrent-dropout", type=float, default=0.0, help="RNN backbone recurrent dropout")
    parser.add_argument(
        "--rnn-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bidirectional GRU in RNN backbone",
    )
    parser.add_argument("--rnn-embedding-dim", type=int, default=512, help="RNN output embedding dimension")
    parser.add_argument("--row", type=int, default=80, help="STFT row/window size")
    parser.add_argument(
        "--enable-ind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable channel-independent transform",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training evaluation")
    parser.add_argument("--knn-k", type=int, default=10, help="K for closed-set KNN evaluation")
    parser.add_argument(
        "--open-set-enroll-k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Number of enrollment probes per open-set UE (can specify multiple for sweep)",
    )
    parser.add_argument(
        "--num-open-set-nodes",
        type=int,
        default=3,
        help="Number of least-populated labels reserved for open-set testing",
    )
    parser.add_argument(
        "--num-known-nodes",
        type=int,
        default=None,
        help="If set, keep only this many known labels (highest sample counts) for training/eval",
    )
    parser.add_argument(
        "--plot-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save confusion matrix and train/test distribution plots",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory for plot outputs (default: <model_path>/plots)",
    )
    parser.add_argument(
        "--top-confusions",
        type=int,
        default=5,
        help="Number of top off-diagonal confusions to print",
    )
    parser.add_argument(
        "--print-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print final train/validation loss",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Local run configuration (edit these variables directly).
    # -------------------------------------------------------------------------

    # Run mode: set exactly one to True.
    RUN_OPEN_SET_ONLY = True      # Load saved model, run open-set eval only
    RUN_TRAINING_AND_TEST = False  # Full training + closed/open-set eval

    hf_repo_id = DEFAULT_HF_REPO
    hf_config_name = DEFAULT_HF_CONFIG
    hf_revision = None
    hf_train_split = "train"
    hf_test_split = "train"

    train_ratio = 0.8
    label_column = "rnti"
    iq_column = "iq"
    rx_ant = 0
    sym_mode = "flatten"
    max_train_samples = None
    max_test_samples = None
    samples_count = 400
    required_iq_len = 39168

    model_path = None
    rx_id = DatasetAPI.RX_1

    batch_size = 32
    loss_type = "triplet_loss"
    backbone = "rnn"          # "cnn" or "rnn"
    alpha = 1.1
    beta = 0.37
    rnn_gru_units = 256
    rnn_num_layers = 2
    rnn_dropout = 0.3
    rnn_recurrent_dropout = 0.0
    rnn_bidirectional = True
    rnn_embedding_dim = 512
    row = 80
    enable_ind = True

    skip_eval = False
    knn_k = 5
    num_open_set_nodes = 10
    num_known_nodes = None
    open_set_enroll_k = [1,3,5,10]

    plot_outputs = True
    plot_dir = None
    top_confusions = 5
    print_history = True

    benchmark_csv = None

    args = SimpleNamespace(
        hf_repo_id=hf_repo_id,
        hf_config_name=hf_config_name,
        hf_revision=hf_revision,
        hf_train_split=hf_train_split,
        hf_test_split=hf_test_split,
        train_ratio=train_ratio,
        label_column=label_column,
        iq_column=iq_column,
        rx_ant=rx_ant,
        sym_mode=sym_mode,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        samples_count=samples_count,
        required_iq_len=required_iq_len,
        model_path=model_path,
        rx_id=rx_id,
        batch_size=batch_size,
        loss_type=loss_type,
        backbone=backbone,
        alpha=alpha,
        beta=beta,
        rnn_gru_units=rnn_gru_units,
        rnn_num_layers=rnn_num_layers,
        rnn_dropout=rnn_dropout,
        rnn_recurrent_dropout=rnn_recurrent_dropout,
        rnn_bidirectional=rnn_bidirectional,
        rnn_embedding_dim=rnn_embedding_dim,
        row=row,
        enable_ind=enable_ind,
        skip_eval=skip_eval,
        knn_k=knn_k,
        num_open_set_nodes=num_open_set_nodes,
        num_known_nodes=num_known_nodes,
        open_set_enroll_k=open_set_enroll_k,
        plot_outputs=plot_outputs,
        plot_dir=plot_dir,
        top_confusions=top_confusions,
        print_history=print_history,
        benchmark_csv=benchmark_csv,
    )

    if RUN_OPEN_SET_ONLY:
        run_open_set_only(args)
    elif RUN_TRAINING_AND_TEST:
        run_training_and_test(args)
