# Device Fingerprinting Tools

This is the crown jewel of this project. This subdirectory contains all the source code related to taking the pre-processed IQ samples, training the fingerprint extractor model, generating device fingerprints, and performing the device search & enrollment tasks.

For detailed description of how this code works, please refer to our paper.

## Status

This fork is now cleaned to focus on **AODT Hugging Face data only** for training/testing TripletNet-based fingerprinting.

## AODT Hugging Face pipeline (TripletNet)

This fork also supports training/testing the fingerprint extractor directly from AODT PUSCH IQ datasets hosted on Hugging Face (as produced by `AODT_IQ_Collection/pusch_iq_dataset/build_hf_dataset.py`).

Use `DatasetAPI.DATASET_AODT_HF` and provide HF config in `data_config`, for example:

```python
data_config = {
    "dataset_name": DatasetAPI.DATASET_AODT_HF,
    "samples_count": 400,
    "hf_repo_id": "your-org/your-aodt-dataset",
    "hf_train_split": "train",
    "hf_test_split": "train",   # same split + ratio split if no test split exists
    "hf_train_ratio": 0.8,
    "hf_label_column": "rnti",
    "hf_rx_ant": 0,
    "hf_sym_mode": "flatten",   # flatten | first_sym | mean_sym
    "model_path": "/tmp/aodt_hf_models",
}
```

Then:
- train with `FingerprintingAPI.train_models(...)`
- test with `EvaluationAPI.evaluate_aodt_hf_closed_set(...)`

Legacy Orbit/WiSig local file loaders and evaluation flows are disabled in this cleaned branch.