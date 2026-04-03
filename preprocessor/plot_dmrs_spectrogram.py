#!/usr/bin/env python3
"""
Plot DMRS channel-response spectrograms from HuggingFace AODT PUSCH datasets.

Python port of plot_dmrs_spectrogram.m and extract_dmrs.m from
powder-srsran-automator/.  Loads IQ samples via parquet, extracts only the
DMRS OFDM symbols (identified from dmrsSymLocBmsk), sorts by (batch, slot)
for correct temporal ordering, and renders spectrogram-style heatmaps.

Axes:
  X  — Subcarrier index within DMRS symbol (frequency).
  Y  — Slot index (time), ordered by batch then slot number.
  Color — Magnitude of the channel response in dB.

Usage examples:
  # Auto-pick the UE with most 39168-length probes in the 200batch config
  python plot_dmrs_spectrogram.py

  # Specify a config and UE explicitly
  python plot_dmrs_spectrogram.py \\
      --hf-config-name data-10UE-1gNB-500batch-10slots-1sample-NoMobility-halfwaveDipole_UE_gNB \\
      --ue-label 6

  # Override output directory
  python plot_dmrs_spectrogram.py --output-dir ./my_plots
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HF_REPO = "CAAI-FAU/PHY-Device-Fingerprinting-cuPHY-PUSCH"
HF_CONFIGS = [
    "data-10UE-1gNB-500batch-10slots-1sample-0.01ms-halfwaveDipole_UE_gNB",
    "data-10UE-1gNB-500batch-10slots-1sample-NoMobility-halfwaveDipole_UE_gNB",
    "data-10UE-1gNB-100batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB",
    "data-10UE-1gNB-200batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB",
    "data-10UE-1gNB-200batch-30slots-1sample-NoMobility-halfwaveDipole_UE_gNB-Hospital",
]
DEFAULT_HF_CONFIG = HF_CONFIGS[3]

META_COLUMNS = [
    "batch", "slot", "rnti", "nSym", "nSc", "sym0", "nPuschSym",
    "dmrsSymLocBmsk", "startPrb", "nPrb", "nRxAnt",
]

# cuPHY RAN configuration for the AODT data collection.
RAN_CONFIG = {
    "fft_size": 4096,
    "scs_hz": 30e3,
    "carrier_bw_hz": 100e6,
    "cyclic_prefix": 288,
    "carrier_freq_hz": 3.6e9,
    "symbols_per_slot": 14,
    "n_subcarriers": 3276,
    "sc_per_prb": 12,
    "n_prb_total": 273,                     # 3276 / 12
    "n_rx_ant": 4,
    "dmrs_positions": [2, 3, 10, 11],
    "slot_duration_ms": 0.5,                # 1 ms / (30 kHz / 15 kHz)
    "gnb_power_dbm": 23,
    "ue_power_dbm": 23,
    "gnb_nf_db": 0.5,
    "ue_nf_db": 0.5,
    "channel_estimation": "MMSE",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dmrs_sym_indices(dmrs_bmsk: int, sym0: int, n_sym: int) -> list[int]:
    """Return local indices (within the stored [n_sym, nSc] block) of DMRS symbols."""
    return [
        s - sym0
        for s in range(14)
        if (dmrs_bmsk >> s) & 1 and sym0 <= s < sym0 + n_sym
    ]


def _parquet_files(repo_id: str, config_name: str) -> list[str]:
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted(
        f for f in all_files
        if f.startswith(config_name + "/train-") and f.endswith(".parquet")
    )


# ---------------------------------------------------------------------------
# Data loading (parquet-based, metadata-aware)
# ---------------------------------------------------------------------------

def load_dmrs_for_device(
    repo_id: str,
    config_name: str,
    rx_ant: int = 0,
    required_iq_len: int = 39168,
    ue_label: int | None = None,
    iq_column: str = "iq",
):
    """Load DMRS-only IQ for a single UE, time-ordered by (batch, slot).

    If *ue_label* is None, automatically picks the UE with the most samples.

    Returns
    -------
    dmrs_matrix : np.ndarray, complex64, [N_slots, n_dmrs_sym * nSc]
    n_dmrs_sym  : int   — number of DMRS symbols per slot
    n_sc        : int   — subcarriers per symbol
    meta_rows   : list[dict] — per-row metadata, time-ordered
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    pfiles = _parquet_files(repo_id, config_name)
    if not pfiles:
        raise RuntimeError(f"No parquet shards for config '{config_name}'")

    # --- Pass 1: read metadata to pick the best UE and collect row indices ---
    shard_locals: list[str] = []
    shard_meta: list[dict] = []
    from collections import Counter
    ue_counts: Counter[int] = Counter()

    for pf in pfiles:
        local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=pf)
        shard_locals.append(local)
        meta = pq.read_table(local, columns=META_COLUMNS, use_threads=True).to_pydict()
        shard_meta.append(meta)
        for i in range(len(meta["rnti"])):
            if meta["nSym"][i] * meta["nSc"][i] != required_iq_len:
                continue
            ue_counts[int(meta["rnti"][i])] += 1

    if not ue_counts:
        raise RuntimeError(
            f"No samples with nSym*nSc={required_iq_len} in config '{config_name}'"
        )

    if ue_label is None:
        ue_label = max(ue_counts, key=ue_counts.get)
    elif ue_label not in ue_counts:
        raise RuntimeError(
            f"UE {ue_label} not found with required_iq_len={required_iq_len}.  "
            f"Available: {dict(sorted(ue_counts.items()))}"
        )

    target_count = ue_counts[ue_label]
    print(f"Selected UE {ue_label} ({target_count} probes with len {required_iq_len})")
    print(f"All UEs with len {required_iq_len}: {dict(sorted(ue_counts.items()))}")

    # --- Pass 2: load IQ rows for the target UE, extract DMRS symbols ---
    rows_collected: list[tuple[int, int, int, np.ndarray, dict]] = []
    dmrs_sym_count = None

    for shard_idx, (local, meta) in enumerate(zip(shard_locals, shard_meta)):
        selected = [
            i for i in range(len(meta["rnti"]))
            if int(meta["rnti"][i]) == ue_label
            and meta["nSym"][i] * meta["nSc"][i] == required_iq_len
        ]
        if not selected:
            continue

        iq_data = pq.read_table(local, columns=[iq_column], use_threads=True) \
                     .to_pydict().get(iq_column, [])

        for i in selected:
            iq_raw = iq_data[i]
            if iq_raw is None:
                continue
            iq_arr = np.asarray(iq_raw, dtype=np.float32)
            if iq_arr.ndim != 3 or iq_arr.shape[-1] % 2 != 0:
                continue

            n_rx = iq_arr.shape[0]
            n_sym = iq_arr.shape[1]
            n_sc = iq_arr.shape[2] // 2
            ant = min(rx_ant, n_rx - 1)

            iq_complex = (
                iq_arr[ant, :, 0::2] + 1j * iq_arr[ant, :, 1::2]
            ).astype(np.complex64)  # [n_sym, n_sc]

            dmrs_idx = _dmrs_sym_indices(
                int(meta["dmrsSymLocBmsk"][i]),
                int(meta["sym0"][i]),
                n_sym,
            )
            if not dmrs_idx:
                continue

            dmrs_iq = iq_complex[dmrs_idx, :]  # [n_dmrs_sym, n_sc]

            if dmrs_sym_count is None:
                dmrs_sym_count = len(dmrs_idx)

            if len(dmrs_idx) != dmrs_sym_count:
                continue

            row_meta = {k: meta[k][i] for k in META_COLUMNS}
            row_meta["dmrs_sym_indices"] = dmrs_idx
            rows_collected.append((
                int(meta["batch"][i]),
                int(meta["slot"][i]),
                i,
                dmrs_iq.reshape(-1),
                row_meta,
            ))

        print(f"  shard {shard_idx+1}/{len(pfiles)}: "
              f"collected {len(rows_collected)} rows so far")

    if not rows_collected:
        raise RuntimeError(f"No valid IQ rows for UE {ue_label}")

    rows_collected.sort(key=lambda x: (x[0], x[1]))

    dmrs_matrix = np.stack([r[3] for r in rows_collected])
    meta_rows = [r[4] for r in rows_collected]
    n_sc = int(meta_rows[0]["nSc"])

    sym0 = int(meta_rows[0]["sym0"])
    actual_dmrs_in_slot = [sym0 + i for i in meta_rows[0]["dmrs_sym_indices"]]
    expected_dmrs = RAN_CONFIG["dmrs_positions"]

    print(
        f"DMRS matrix: [{dmrs_matrix.shape[0]} slots × {dmrs_matrix.shape[1]} values] "
        f"({dmrs_sym_count} DMRS syms × {n_sc} sc)"
    )
    print(f"DMRS symbol positions in slot: {actual_dmrs_in_slot}")
    if actual_dmrs_in_slot != expected_dmrs:
        print(f"  [WARN] Expected DMRS at {expected_dmrs} from RAN config")
    else:
        print(f"  [OK] Matches RAN config DMRS positions {expected_dmrs}")
    print(f"PRB allocation: startPrb={meta_rows[0]['startPrb']}, nPrb={meta_rows[0]['nPrb']}")

    return dmrs_matrix, dmrs_sym_count, n_sc, meta_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _short_cfg(config_name: str) -> str:
    return config_name.replace("data-", "", 1)


def plot_combined(
    dmrs_matrix: np.ndarray,
    n_dmrs_sym: int,
    n_sc: int,
    meta_rows: list[dict],
    output_path: str,
    config_name: str = "",
):
    """Single-panel heatmap with DMRS symbols concatenated on the x-axis."""
    n_slots, n_total = dmrs_matrix.shape
    ue_label = meta_rows[0]["rnti"]
    n_prb = meta_rows[0]["nPrb"]
    dmrs_sym_idx = meta_rows[0]["dmrs_sym_indices"]
    scs_khz = RAN_CONFIG["scs_hz"] / 1e3

    slot_ms = RAN_CONFIG["slot_duration_ms"]
    batches = np.array([m["batch"] for m in meta_rows])
    time_ms = np.arange(n_slots) * slot_ms

    full_mhz = np.arange(n_total) * scs_khz / 1e3
    bw_mhz = n_sc * scs_khz / 1e3
    fc_ghz = RAN_CONFIG["carrier_freq_hz"] / 1e9

    m_db = 20 * np.log10(np.abs(dmrs_matrix) + np.finfo(np.float32).eps)
    clim = (np.percentile(m_db, 2), np.percentile(m_db, 98))

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        m_db, aspect="auto", origin="lower",
        extent=[0, full_mhz[-1], time_ms[0], time_ms[-1]],
        vmin=clim[0], vmax=clim[1], cmap="jet", interpolation="nearest",
    )
    for s in range(1, n_dmrs_sym):
        ax.axvline(s * bw_mhz, color="white", lw=0.6, ls="--", alpha=0.7)

    ax.set_xlabel("Subcarrier offset within DMRS symbol (MHz)")
    ax.set_ylabel("Time (ms)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("|H$_{rx}$ / H$_{ideal}$| (dB)")

    sym0 = meta_rows[0]["sym0"]
    slot_syms = [sym0 + i for i in dmrs_sym_idx]
    ax.set_title(
        f"UE {ue_label}  |  {n_slots} slots × {n_total} DMRS subcarriers  "
        f"({n_dmrs_sym} DMRS syms × {n_sc} sc)\n"
        f"f$_c$={fc_ghz} GHz  |  BW={RAN_CONFIG['carrier_bw_hz']/1e6:.0f} MHz  |  "
        f"SCS={scs_khz:.0f} kHz  |  {n_prb} PRBs  |  "
        f"DMRS syms {RAN_CONFIG['dmrs_positions']}  |  "
        f"batches {batches[0]}–{batches[-1]}",
        fontsize=10,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_symbol(
    dmrs_matrix: np.ndarray,
    n_dmrs_sym: int,
    n_sc: int,
    meta_rows: list[dict],
    output_path: str,
    config_name: str = "",
):
    """Single row of panels, one per DMRS symbol.

    Layout closely follows the MATLAB plot_dmrs_spectrogram.m reference:
    wide landscape figure, tall panels side-by-side, shared colorbar on the
    east edge, Y = Time (ms), X = Subcarrier offset (MHz).
    """
    n_slots, n_total = dmrs_matrix.shape
    ue_label = meta_rows[0]["rnti"]
    n_prb = meta_rows[0]["nPrb"]
    dmrs_sym_idx = meta_rows[0]["dmrs_sym_indices"]
    sym0 = meta_rows[0]["sym0"]
    slot_syms = [sym0 + i for i in dmrs_sym_idx]

    scs_khz = RAN_CONFIG["scs_hz"] / 1e3
    slot_ms = RAN_CONFIG["slot_duration_ms"]
    fc_ghz = RAN_CONFIG["carrier_freq_hz"] / 1e9
    time_ms = np.arange(n_slots) * slot_ms
    sc_mhz = np.arange(n_sc) * scs_khz / 1e3
    bw_mhz = n_sc * scs_khz / 1e3

    m_db = 20 * np.log10(np.abs(dmrs_matrix) + np.finfo(np.float32).eps)
    clim = (np.percentile(m_db, 2), np.percentile(m_db, 98))

    panel_w = 3.5
    fig_w = panel_w * n_dmrs_sym + 1.6
    fig_h = 8
    fig, axes = plt.subplots(
        1, n_dmrs_sym,
        figsize=(fig_w, fig_h),
        sharey=True,
    )
    if n_dmrs_sym == 1:
        axes = [axes]

    for s, ax in enumerate(axes):
        block = m_db[:, s * n_sc : (s + 1) * n_sc]
        im = ax.imshow(
            block, aspect="auto", origin="lower",
            extent=[sc_mhz[0], sc_mhz[-1], time_ms[0], time_ms[-1]],
            vmin=clim[0], vmax=clim[1], cmap="jet", interpolation="nearest",
        )
        ax.set_title(
            f"UE {ue_label} — DMRS Symbol {slot_syms[s]}",
            fontsize=10,
        )
        ax.set_xlabel("Subcarrier offset within DMRS symbol (MHz)")
        if s == 0:
            ax.set_ylabel("Time (ms)")

    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, shrink=0.9)
    cb.set_label("|H$_{rx}$/H$_{ideal}$| (dB)")

    fig.suptitle(
        f"UE {ue_label}  |  {n_slots} slots × {n_total} DMRS subcarriers\n"
        f"f$_c$={fc_ghz} GHz  |  BW={RAN_CONFIG['carrier_bw_hz']/1e6:.0f} MHz  "
        f"|  SCS={scs_khz:.0f} kHz  |  {n_prb}/{RAN_CONFIG['n_prb_total']} PRBs  "
        f"|  {RAN_CONFIG['channel_estimation']} ch. est.  "
        f"|  Colour = channel+HW response (RF fingerprint)",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.06, right=0.91, bottom=0.08, top=0.90,
        wspace=0.08,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot DMRS spectrograms from HF AODT PUSCH datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--hf-repo-id", default=HF_REPO)
    p.add_argument("--hf-config-name", default=DEFAULT_HF_CONFIG,
                    help="HF dataset config name")
    p.add_argument("--rx-ant", type=int, default=0, help="RX antenna index")
    p.add_argument("--required-iq-len", type=int, default=39168,
                    help="Keep only samples with nSym*nSc equal to this")
    p.add_argument("--ue-label", type=int, default=None,
                    help="Device RNTI to plot (default: auto-pick UE with most data)")
    p.add_argument("--iq-column", default="iq")
    p.add_argument("--output-dir", default=None,
                    help="Output directory (default: aodt_hf_models/plots)")
    p.add_argument("--per-symbol", action="store_true",
                    help="Also produce per-DMRS-symbol grid plot")
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(repo_root, "aodt_hf_models", "plots")
    os.makedirs(output_dir, exist_ok=True)

    scs_khz = RAN_CONFIG["scs_hz"] / 1e3
    print("=== DMRS Spectrogram Plotter ===")
    print(f"Repo:   {args.hf_repo_id}")
    print(f"Config: {args.hf_config_name}")
    print(f"RX ant: {args.rx_ant}")
    print(f"Required IQ len: {args.required_iq_len}")
    if args.ue_label is not None:
        print(f"UE filter: {args.ue_label}")
    else:
        print("UE filter: auto (most data)")
    print()
    print("--- RAN Configuration ---")
    print(f"  Carrier freq:  {RAN_CONFIG['carrier_freq_hz']/1e9:.1f} GHz")
    print(f"  Bandwidth:     {RAN_CONFIG['carrier_bw_hz']/1e6:.0f} MHz")
    print(f"  SCS:           {scs_khz:.0f} kHz")
    print(f"  FFT size:      {RAN_CONFIG['fft_size']}")
    print(f"  Subcarriers:   {RAN_CONFIG['n_subcarriers']} ({RAN_CONFIG['n_prb_total']} PRBs)")
    print(f"  Slot duration: {RAN_CONFIG['slot_duration_ms']} ms")
    print(f"  DMRS symbols:  {RAN_CONFIG['dmrs_positions']}")
    print(f"  RX antennas:   {RAN_CONFIG['n_rx_ant']}")
    print(f"  Ch. est.:      {RAN_CONFIG['channel_estimation']}")
    print()

    dmrs_matrix, n_dmrs_sym, n_sc, meta_rows = load_dmrs_for_device(
        repo_id=args.hf_repo_id,
        config_name=args.hf_config_name,
        rx_ant=args.rx_ant,
        required_iq_len=args.required_iq_len,
        ue_label=args.ue_label,
        iq_column=args.iq_column,
    )

    ue = meta_rows[0]["rnti"]
    cfg_short = args.hf_config_name.replace("data-", "").replace(
        "-halfwaveDipole_UE_gNB", ""
    )

    combined_path = os.path.join(output_dir, f"dmrs_spectrogram_ue{ue}_{cfg_short}.png")
    plot_combined(
        dmrs_matrix, n_dmrs_sym, n_sc, meta_rows, combined_path,
        config_name=args.hf_config_name,
    )

    if args.per_symbol:
        sym_path = os.path.join(
            output_dir, f"dmrs_spectrogram_per_sym_ue{ue}_{cfg_short}.png",
        )
        plot_per_symbol(
            dmrs_matrix, n_dmrs_sym, n_sc, meta_rows, sym_path,
            config_name=args.hf_config_name,
        )

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
