import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py


def compute_channel_wise_mean_and_std(file_paths: list):
    """
    Calculates the channel-wise mean and standard deviation for a list of HDF5 files.
    """

    # Get the number of channels from the first file
    with h5py.File(file_paths[0], 'r') as f:
        num_channels = f['data'].shape[0]

    # Use float64 for accumulators to avoid precision loss with many additions
    channel_sum = np.zeros(num_channels, dtype=np.float64)
    channel_sum_squared = np.zeros(num_channels, dtype=np.float64)
    num_files = len(file_paths)

    for file_path in tqdm(file_paths, desc="Calculating Stats"):
        with h5py.File(file_path, 'r') as f:
            # Load data as float64 to contribute to the high-precision sum
            data = f['data'][:].astype(np.float64)

        channel_sum += data.mean(axis=(1, 2))
        channel_sum_squared += (data ** 2).mean(axis=(1, 2))

    mean = channel_sum / num_files
    # E[X^2] - (E[X])^2. Add epsilon for numerical stability.
    std = np.sqrt((channel_sum_squared / num_files) - mean ** 2 + 1e-8)

    # Return as float32
    return mean.astype(np.float32), std.astype(np.float32)


FOLDS_CSV_PATH = Path("./data/10fold_train_val_subjects.csv")
WSST_CSV_PATH = Path("./data/wsst.csv")
CWT_CSV_PATH = Path("./data/cwt.csv")
OUTPUT_DIR = Path("./data/normalization")

print("Loading data files...")
folds_data = pd.read_csv(FOLDS_CSV_PATH)
wsst_df = pd.read_csv(WSST_CSV_PATH)
cwt_df = pd.read_csv(CWT_CSV_PATH)

print("Preprocessing DataFrames for fast lookup...")
cwt_df['subject_folder'] = cwt_df['file_path'].apply(lambda p: Path(p).parent.name)
wsst_df['subject_folder'] = wsst_df['file_path'].apply(lambda p: Path(p).parent.name)

data_to_process = {
    'cwt': cwt_df,
    'wsst': wsst_df
}

for data_name, df in data_to_process.items():
    print(f"\n--- Processing data type: {data_name.upper()} ---")

    for fold_idx in range(1, 11):
        train_subjects = folds_data[
            (folds_data['fold'] == fold_idx) & (folds_data['split'] == 'train')
            ]['subject_folder'].tolist()

        train_df_fold = df[df['subject_folder'].isin(train_subjects)]
        train_file_paths = train_df_fold['file_path'].tolist()

        if not train_file_paths:
            print(f"Warning: No training files found for {data_name} in fold {fold_idx}. Skipping.")
            continue

        print(f"Calculating stats for fold {fold_idx} with {len(train_file_paths)} files...")

        fold_mean, fold_std = compute_channel_wise_mean_and_std(train_file_paths)

        if fold_mean is not None and fold_std is not None:
            np.save(OUTPUT_DIR / f'{data_name}_mean_fold{fold_idx}', fold_mean)
            np.save(OUTPUT_DIR / f'{data_name}_std_fold{fold_idx}', fold_std)
            print(f"Saved stats for {data_name} fold {fold_idx}.")

print("\nAll normalization statistics have been computed and saved.")