import numpy as np
from pathlib import Path
from datasets import load_dataset, Sequence, Value
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None
try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    hf_hub_download = None
    list_repo_files = None

np.random.seed(42)

class DatasetAPI():

    DATASET_AODT_HF = 'aodt_hf' # Hugging Face dataset built from AODT PUSCH IQ collection

    RX_1 = 'node1-1'
    RX_2 = 'node1-20'
    RX_3 = 'node20-1'
    RX_4 = 'node19-19'

    def __init__(self, root_dir, matlab_src_dir, matlab_session_id, aug_on=False, seed=42):
        self.rng = np.random.RandomState(seed)
        self.root_dir = root_dir

    def _shuffle_dataset(self, data, labels, rssi):
        # Produce a new order for elements
        new_order = np.arange(labels.shape[0])
        self.rng.shuffle(new_order)

        data = data[new_order, :]
        labels = labels[new_order]
        rssi = rssi[new_order] if rssi is not None else None

        return data, labels, rssi

    def _require_hf_datasets(self):
        if load_dataset is None or Sequence is None or Value is None:
            raise RuntimeError(
                "Missing dependency `datasets`. Install it with `pip install datasets`."
            )

    def _normalize_filter_values(self, values):
        if values is None:
            return None
        return set(int(v) for v in values)

    def _load_hf_dataset_from_parquet(
        self,
        repo_id,
        split='train',
        revision=None,
        config_name=None,
        label_column='rnti',
        iq_column='iq',
        rx_ant=0,
        sym_mode='flatten',
        batch_filter=None,
        slot_filter=None,
        max_samples=None,
        required_iq_len=None,
    ):
        if pq is None or hf_hub_download is None or list_repo_files is None:
            raise RuntimeError("Parquet loader requires `pyarrow` and `huggingface_hub`.")

        repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
        if config_name:
            prefix_candidates = [f"{config_name}/{split}-", f"{config_name}/data/{split}-"]
        else:
            prefix_candidates = [f"data/{split}-"]
        parquet_files = sorted(
            [f for f in repo_files if f.endswith(".parquet") and any(f.startswith(pfx) for pfx in prefix_candidates)]
        )
        if not parquet_files:
            raise RuntimeError(
                f"No parquet shards found for split='{split}' in dataset '{repo_id}' "
                f"(config='{config_name}')."
            )

        frames = []
        labels = []
        rssis = []
        frame_lengths = []
        skipped_len_mismatch = 0
        t0 = None

        for shard in parquet_files:
            local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", revision=revision, filename=shard)

            schema_names = set(pq.read_schema(local_path).names)
            meta_cols = [label_column, 'batch', 'slot', 'nSym', 'nSc']
            if 'rssi' in schema_names:
                meta_cols.append('rssi')
            meta = pq.read_table(local_path, columns=meta_cols, use_threads=True).to_pydict()
            if label_column not in meta:
                continue

            row_count = len(meta[label_column])
            selected_idx = []
            for i in range(row_count):
                if batch_filter is not None and int(meta['batch'][i]) not in batch_filter:
                    continue
                if slot_filter is not None and int(meta['slot'][i]) not in slot_filter:
                    continue
                if meta[label_column][i] is None:
                    continue

                if required_iq_len is not None:
                    n_sym = int(meta['nSym'][i] or 0)
                    n_sc = int(meta['nSc'][i] or 0)
                    if (n_sym <= 0) or (n_sc <= 0) or (n_sym * n_sc != int(required_iq_len)):
                        skipped_len_mismatch += 1
                        continue

                selected_idx.append(i)

            if not selected_idx:
                continue

            iq_data = pq.read_table(local_path, columns=[iq_column], use_threads=True).to_pydict().get(iq_column, [])
            for i in selected_idx:
                if max_samples is not None and len(frames) >= max_samples:
                    break

                iq_value = iq_data[i] if i < len(iq_data) else None
                if iq_value is None:
                    continue

                iq = self._hf_iq_to_1d_complex(iq_value, rx_ant=rx_ant, sym_mode=sym_mode)
                if iq is None:
                    continue

                try:
                    label_val = int(meta[label_column][i])
                except Exception:
                    continue

                frames.append(iq)
                labels.append(label_val)
                if 'rssi' in meta:
                    rssis.append(float(meta['rssi'][i]) if (meta['rssi'][i] is not None) else np.nan)
                else:
                    rssis.append(np.nan)
                frame_lengths.append(iq.shape[0])

            if max_samples is not None and len(frames) >= max_samples:
                break

        if not frames:
            raise RuntimeError(
                f"No usable records loaded from HF parquet dataset '{repo_id}' (split='{split}'). "
                "Check filters and required_iq_len."
            )

        if required_iq_len is not None:
            print(
                f"[INFO] Enforced required IQ length={int(required_iq_len)}. "
                f"Kept={len(frames)} samples, dropped={skipped_len_mismatch} length-mismatched samples."
            )

        min_len = int(np.min(frame_lengths))
        if len(set(frame_lengths)) > 1:
            print(f"[WARN] Variable IQ lengths in HF data. Truncating all frames to min length={min_len}.")
        frames = np.asarray([x[:min_len] for x in frames], dtype=np.complex64)
        labels = np.asarray(labels, dtype=int).reshape(-1, 1)
        if np.isnan(rssis).all():
            rssis = None
        else:
            rssis = np.asarray(rssis, dtype=np.float32).reshape(-1, 1)
        return frames, labels, rssis

    def _load_hf_iq_from_path(self, row):
        iq_path = row.get('iq_path')
        if iq_path is None:
            return None

        n_rx_ant = int(row.get('nRxAnt', 0) or 0)
        n_sym = int(row.get('nSym', 0) or 0)
        n_sc = int(row.get('nSc', 0) or 0)
        if n_rx_ant <= 0 or n_sym <= 0 or n_sc <= 0:
            return None

        iq_path = Path(iq_path)
        if not iq_path.exists():
            return None

        raw = np.fromfile(iq_path, dtype=np.float32)
        expected = 2 * n_rx_ant * n_sym * n_sc
        if raw.size != expected:
            return None

        iq = raw.reshape(n_rx_ant, n_sym, n_sc * 2)
        return iq

    def _hf_iq_to_1d_complex(self, iq_interleaved, rx_ant=0, sym_mode='flatten'):
        iq_arr = np.asarray(iq_interleaved, dtype=np.float32)
        if iq_arr.ndim != 3 or iq_arr.shape[-1] % 2 != 0:
            return None

        n_rx_ant = iq_arr.shape[0]
        rx_ant = int(np.clip(rx_ant, 0, n_rx_ant - 1))

        i = iq_arr[rx_ant, :, 0::2]
        q = iq_arr[rx_ant, :, 1::2]
        iq_complex = (i + 1j * q).astype(np.complex64, copy=False)

        if sym_mode == 'first_sym':
            return iq_complex[0, :]
        if sym_mode == 'mean_sym':
            return iq_complex.mean(axis=0)
        return iq_complex.reshape(-1)

    def _split_by_device_ratio(self, data, labels, rssi, train_ratio=0.8):
        label_values = labels.flatten().astype(int)
        unique_labels = sorted(set(label_values))

        train_idx = []
        test_idx = []
        for dev in unique_labels:
            dev_idx = np.where(label_values == dev)[0]
            self.rng.shuffle(dev_idx)
            split_i = int(len(dev_idx) * train_ratio)
            split_i = min(max(split_i, 1), len(dev_idx) - 1) if len(dev_idx) > 1 else len(dev_idx)
            train_idx.extend(dev_idx[:split_i])
            test_idx.extend(dev_idx[split_i:])

        train_idx = np.array(train_idx, dtype=int)
        test_idx = np.array(test_idx, dtype=int)

        data_train = data[train_idx]
        labels_train = labels[train_idx]
        rssi_train = rssi[train_idx] if rssi is not None else None

        data_test = data[test_idx]
        labels_test = labels[test_idx]
        rssi_test = rssi[test_idx] if rssi is not None else None

        return data_train, labels_train, rssi_train, data_test, labels_test, rssi_test

    def load_hf_dataset(
        self,
        repo_id,
        split='train',
        revision=None,
        config_name=None,
        label_column='rnti',
        iq_column='iq',
        rx_ant=0,
        sym_mode='flatten',
        batch_filter=None,
        slot_filter=None,
        max_samples=None,
        shuffle=False,
        required_iq_len=None,
        prefer_parquet_loader=False,
    ):
        """
        Load AODT IQ records from a Hugging Face dataset.
        Expected IQ schema is [nRxAnt, nSym, 2*nSc] with interleaved I/Q.
        """
        if prefer_parquet_loader:
            try:
                frames, labels, rssis = self._load_hf_dataset_from_parquet(
                    repo_id=repo_id,
                    split=split,
                    revision=revision,
                    config_name=config_name,
                    label_column=label_column,
                    iq_column=iq_column,
                    rx_ant=rx_ant,
                    sym_mode=sym_mode,
                    batch_filter=batch_filter,
                    slot_filter=slot_filter,
                    max_samples=max_samples,
                    required_iq_len=required_iq_len,
                )
                if shuffle:
                    frames, labels, rssis = self._shuffle_dataset(frames, labels, rssis)
                return frames, labels, rssis
            except Exception as e:
                print(f"[WARN] Parquet loader failed; falling back to datasets streaming: {e}")

        self._require_hf_datasets()

        batch_filter = self._normalize_filter_values(batch_filter)
        slot_filter = self._normalize_filter_values(slot_filter)

        # Streaming avoids loading the full table into memory.
        ds = load_dataset(repo_id, name=config_name, split=split, revision=revision, streaming=True)
        # Some rows have variable IQ width; cast to variable-length nested
        # sequences to avoid Array3D reshape failures at iteration time.
        if hasattr(ds, "features") and iq_column in ds.features:
            features = ds.features.copy()
            features[iq_column] = Sequence(Sequence(Sequence(Value("float32"))))
            ds = ds.cast(features)

        frames = []
        labels = []
        rssis = []
        frame_lengths = []
        skipped_len_mismatch = 0

        scanned_rows = 0
        t0 = np.datetime64('now')
        for row in ds:
            scanned_rows += 1
            if batch_filter is not None and int(row.get('batch', -1)) not in batch_filter:
                continue
            if slot_filter is not None and int(row.get('slot', -1)) not in slot_filter:
                continue

            if label_column not in row or row[label_column] is None:
                continue

            iq_value = row.get(iq_column)
            if iq_value is None:
                iq_value = self._load_hf_iq_from_path(row)
            if iq_value is None:
                continue

            iq = self._hf_iq_to_1d_complex(iq_value, rx_ant=rx_ant, sym_mode=sym_mode)
            if iq is None:
                continue
            if required_iq_len is not None and int(iq.shape[0]) != int(required_iq_len):
                skipped_len_mismatch += 1
                continue

            try:
                label_val = int(row[label_column])
            except Exception:
                continue

            frames.append(iq)
            labels.append(label_val)
            rssis.append(float(row['rssi']) if ('rssi' in row and row['rssi'] is not None) else np.nan)
            frame_lengths.append(iq.shape[0])

            if max_samples is not None and len(frames) >= max_samples:
                break
            if scanned_rows % 500 == 0:
                elapsed_s = (np.datetime64('now') - t0) / np.timedelta64(1, 's')
                print(
                    f"[INFO] HF load progress split='{split}': "
                    f"scanned={scanned_rows}, kept={len(frames)}, elapsed={float(elapsed_s):.1f}s"
                )

        if not frames:
            raise RuntimeError(
                f"No usable records loaded from HF dataset '{repo_id}' (split='{split}'). "
                "Check filters, columns, and whether IQ is embedded."
            )
        if required_iq_len is not None:
            print(
                f"[INFO] Enforced required IQ length={int(required_iq_len)}. "
                f"Kept={len(frames)} samples, dropped={skipped_len_mismatch} length-mismatched samples."
            )

        # Keep a fixed frame length expected by downstream models.
        min_len = int(np.min(frame_lengths))
        if len(set(frame_lengths)) > 1:
            print(f"[WARN] Variable IQ lengths in HF data. Truncating all frames to min length={min_len}.")
        frames = np.asarray([x[:min_len] for x in frames], dtype=np.complex64)
        labels = np.asarray(labels, dtype=int).reshape(-1, 1)

        if np.isnan(rssis).all():
            rssis = None
        else:
            rssis = np.asarray(rssis, dtype=np.float32).reshape(-1, 1)

        if shuffle:
            frames, labels, rssis = self._shuffle_dataset(frames, labels, rssis)

        return frames, labels, rssis

    def load_hf_train_test(self, data_config, shuffle_train=True, shuffle_test=False):
        if data_config.get('dataset_name') != self.DATASET_AODT_HF:
            raise ValueError("load_hf_train_test requires dataset_name='aodt_hf'")

        repo_id = data_config.get('hf_repo_id')
        if not repo_id:
            raise ValueError("Missing `hf_repo_id` in data_config for AODT HF dataset.")

        revision = data_config.get('hf_revision', None)
        config_name = data_config.get('hf_config_name', None)
        train_split = data_config.get('hf_train_split', 'train')
        test_split = data_config.get('hf_test_split', train_split)
        label_column = data_config.get('hf_label_column', 'rnti')
        iq_column = data_config.get('hf_iq_column', 'iq')
        rx_ant = data_config.get('hf_rx_ant', 0)
        sym_mode = data_config.get('hf_sym_mode', 'flatten')
        required_iq_len = data_config.get('hf_required_iq_len', None)

        data_train, labels_train, rssi_train = self.load_hf_dataset(
            repo_id=repo_id,
            split=train_split,
            revision=revision,
            config_name=config_name,
            label_column=label_column,
            iq_column=iq_column,
            rx_ant=rx_ant,
            sym_mode=sym_mode,
            batch_filter=data_config.get('hf_train_batches', None),
            slot_filter=data_config.get('hf_train_slots', None),
            max_samples=data_config.get('hf_max_train_samples', None),
            shuffle=shuffle_train if train_split != test_split else False,
            required_iq_len=required_iq_len,
        )

        if train_split == test_split and not data_config.get('hf_test_batches') and not data_config.get('hf_test_slots'):
            ratio = float(data_config.get('hf_train_ratio', 0.8))
            ratio = min(max(ratio, 0.05), 0.95)
            (
                data_train,
                labels_train,
                rssi_train,
                data_test,
                labels_test,
                rssi_test,
            ) = self._split_by_device_ratio(data_train, labels_train, rssi_train, train_ratio=ratio)
            if shuffle_train:
                data_train, labels_train, rssi_train = self._shuffle_dataset(data_train, labels_train, rssi_train)
            if shuffle_test:
                data_test, labels_test, rssi_test = self._shuffle_dataset(data_test, labels_test, rssi_test)
        else:
            data_test, labels_test, rssi_test = self.load_hf_dataset(
                repo_id=repo_id,
                split=test_split,
                revision=revision,
                config_name=config_name,
                label_column=label_column,
                iq_column=iq_column,
                rx_ant=rx_ant,
                sym_mode=sym_mode,
                batch_filter=data_config.get('hf_test_batches', None),
                slot_filter=data_config.get('hf_test_slots', None),
                max_samples=data_config.get('hf_max_test_samples', None),
                shuffle=shuffle_test,
                required_iq_len=required_iq_len,
            )

        node_ids_train = sorted(list(set(labels_train.flatten().astype(int))))
        node_ids_test = sorted(list(set(labels_test.flatten().astype(int))))

        return data_train, labels_train, rssi_train, data_test, labels_test, rssi_test, node_ids_train, node_ids_test

    def filter_dataset(self, data, labels, rssi, dev_range, pkt_range):
        # If the list of devices isn't specified - loop through all of the available ones
        if dev_range is None:
            dev_range = set(labels.flatten())

        try:
            # Filter indexes of frames to keep based on dev_range
            frame_idx_filtered = []
            for dev_idx in dev_range:
                # Find indexes of frames from a specified device
                frame_idx_device = np.where(labels.flatten()==int(dev_idx))[0]
                # Keep a specified sub-range of frames
                frame_idx_device = frame_idx_device[pkt_range]
                # Add these label indexes to our combined array
                frame_idx_filtered.extend(frame_idx_device)
            
            # Filter the dataset based on dev_range and pkt_range
            labels = labels[frame_idx_filtered]
            rssi = rssi[frame_idx_filtered] if rssi is not None else None
            data = data[frame_idx_filtered, :]

            return data, labels, rssi
        except Exception as e:
            return None, None, None

    def rssi_to_weight(self, rssi_dbm):
        # Convert RSSI to weighting factor by normalizing between [0, 1]
        # Note: weak signal is below 90 dBm, very strong signal is above -10 dBm
        if rssi_dbm < -100: 
            print(f'RSSI: {rssi_dbm}. Adjust normalization!')

        rssi_bounds = [-100, 0]
        rssi_scaled = (rssi_dbm - rssi_bounds[0]) / (rssi_bounds[1] - rssi_bounds[0])

        return rssi_scaled

if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")