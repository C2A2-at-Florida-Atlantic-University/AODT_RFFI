import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


HF_REPO_ID = "CAAI-FAU/PHY-Device-Fingerprinting-cuPHY-PUSCH"
HF_CONFIGS = [
    # "data-10UE-1gNB-500batch-10slots-1sample-0.01ms-halfwaveDipole_UE_gNB",
    # "data-10UE-1gNB-500batch-10slots-1sample-NoMobility-halfwaveDipole_UE_gNB",
    # "data-10UE-1gNB-100batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB",
    # "data-10UE-1gNB-200batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB",
    "data-10UE-1gNB-200batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB-Hospital",
]
OUTPUT_DIR = "/home/Research/AODT_RFFI/aodt_hf_models/dataset_analysis"
MIN_SAMPLES_PER_UE = 20


def _safe_name(value):
    chars = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)


def _get_parquet_files(repo_id, config_name):
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted([f for f in files if f.startswith(f"{config_name}/train-") and f.endswith(".parquet")])


def _load_label_and_length(repo_id, config_name):
    parquet_files = _get_parquet_files(repo_id, config_name)
    if not parquet_files:
        raise RuntimeError(f"No parquet files found for config '{config_name}'")

    labels = []
    lengths = []
    for rel_path in parquet_files:
        local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_path)
        table = pq.read_table(local, columns=["rnti", "nSym", "nSc"], use_threads=True)
        d = table.to_pydict()
        rnti = np.asarray(d["rnti"], dtype=np.int32)
        n_sym = np.asarray(d["nSym"], dtype=np.int32)
        n_sc = np.asarray(d["nSc"], dtype=np.int32)
        labels.append(rnti)
        lengths.append(n_sym.astype(np.int64) * n_sc.astype(np.int64))

    labels = np.concatenate(labels, axis=0)
    lengths = np.concatenate(lengths, axis=0)
    return labels, lengths


def _retention_curve(lengths, thresholds):
    ret = []
    for t in thresholds:
        keep = int(np.sum(lengths == t)) if False else int(np.sum(lengths >= t))
        ret.append(keep / len(lengths))
    return np.asarray(ret, dtype=np.float32)


def _best_threshold_all_ues(labels, lengths, min_samples_per_ue):
    unique_labels = sorted(np.unique(labels).tolist())
    candidate_thresholds = sorted(np.unique(lengths).tolist())
    best = None
    for thr in candidate_thresholds:
        ok = True
        for ue in unique_labels:
            cnt = int(np.sum((labels == ue) & (lengths >= thr)))
            if cnt < min_samples_per_ue:
                ok = False
                break
        if ok:
            best = thr
    return best


def _plot_single_config(config_name, labels, lengths, output_dir):
    unique_labels = sorted(np.unique(labels).tolist())
    counts_per_ue = [int(np.sum(labels == ue)) for ue in unique_labels]

    threshold_grid = np.unique(lengths)
    threshold_grid.sort()
    retention = _retention_curve(lengths, threshold_grid)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=140)

    # Samples per UE
    axes[0, 0].bar([str(x) for x in unique_labels], counts_per_ue, color="#1f77b4")
    axes[0, 0].set_title("Samples per UE")
    axes[0, 0].set_xlabel("UE label")
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate(counts_per_ue):
        axes[0, 0].text(i, v + max(1, int(max(counts_per_ue) * 0.01)), str(v), ha="center", va="bottom", fontsize=8)

    # IQ length histogram
    axes[0, 1].hist(lengths, bins=min(40, max(10, len(np.unique(lengths)))), color="#ff7f0e", edgecolor="black")
    axes[0, 1].set_title("IQ length distribution (nSym*nSc)")
    axes[0, 1].set_xlabel("IQ length")
    axes[0, 1].set_ylabel("Frequency")

    # Retention curve
    axes[1, 0].plot(threshold_grid, retention * 100.0, color="#2ca02c", linewidth=2)
    axes[1, 0].set_title("Retention vs required_iq_len")
    axes[1, 0].set_xlabel("required_iq_len")
    axes[1, 0].set_ylabel("Retained samples (%)")
    axes[1, 0].grid(True, alpha=0.3)

    # Per-UE median/min/max IQ length
    ue_min = [int(np.min(lengths[labels == ue])) for ue in unique_labels]
    ue_med = [int(np.median(lengths[labels == ue])) for ue in unique_labels]
    ue_max = [int(np.max(lengths[labels == ue])) for ue in unique_labels]
    x = np.arange(len(unique_labels))
    axes[1, 1].plot(x, ue_min, label="min", marker="o")
    axes[1, 1].plot(x, ue_med, label="median", marker="o")
    axes[1, 1].plot(x, ue_max, label="max", marker="o")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(u) for u in unique_labels])
    axes[1, 1].set_title("Per-UE IQ length stats")
    axes[1, 1].set_xlabel("UE label")
    axes[1, 1].set_ylabel("IQ length")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Dataset analysis: {config_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(output_dir, f"{_safe_name(config_name)}_analysis.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_cross_config_retention(config_to_lengths, output_dir):
    fig = plt.figure(figsize=(10, 6), dpi=140)
    for config_name, lengths in config_to_lengths.items():
        thr = np.unique(lengths)
        thr.sort()
        ret = _retention_curve(lengths, thr) * 100.0
        plt.plot(thr, ret, linewidth=2, label=config_name)
    plt.xlabel("required_iq_len")
    plt.ylabel("Retained samples (%)")
    plt.title("Retention curves across configs")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    out_path = os.path.join(output_dir, "cross_config_retention.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_rows = []
    config_to_lengths = {}

    print("=== HF Dataset Analysis ===")
    print(f"Repo: {HF_REPO_ID}")
    print(f"Configs: {HF_CONFIGS}")
    print()

    for cfg in HF_CONFIGS:
        labels, lengths = _load_label_and_length(HF_REPO_ID, cfg)
        config_to_lengths[cfg] = lengths
        ue_count = int(len(np.unique(labels)))
        n = int(len(lengths))
        best_thr = _best_threshold_all_ues(labels, lengths, MIN_SAMPLES_PER_UE)
        plot_path = _plot_single_config(cfg, labels, lengths, OUTPUT_DIR)

        row = {
            "config": cfg,
            "samples": n,
            "ues": ue_count,
            "min_iq_len": int(np.min(lengths)),
            "p25_iq_len": int(np.percentile(lengths, 25)),
            "median_iq_len": int(np.median(lengths)),
            "p75_iq_len": int(np.percentile(lengths, 75)),
            "max_iq_len": int(np.max(lengths)),
            "best_required_iq_len_all_ues_min_samples_20": int(best_thr) if best_thr is not None else None,
            "plot": plot_path,
        }
        summary_rows.append(row)

        print(f"[{cfg}] samples={n}, ues={ue_count}, min={row['min_iq_len']}, median={row['median_iq_len']}, max={row['max_iq_len']}")
        print(f"  suggested required_iq_len (all UEs keep >= {MIN_SAMPLES_PER_UE} samples): {row['best_required_iq_len_all_ues_min_samples_20']}")
        print(f"  plot: {plot_path}")
        print()

    cross_plot = _plot_cross_config_retention(config_to_lengths, OUTPUT_DIR)
    print(f"Cross-config retention plot: {cross_plot}")

    csv_path = os.path.join(OUTPUT_DIR, "dataset_summary.csv")
    header = list(summary_rows[0].keys()) if summary_rows else []
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in summary_rows:
            f.write(",".join([str(row[k]) for k in header]) + "\n")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
