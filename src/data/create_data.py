import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import cv2
from ssqueezepy import ssq_cwt
from ssqueezepy.wavelets import Wavelet


os.environ['SSQ_GPU'] = '1'

ROOT_DIRECTORY = Path('./data/sleep-edf-database-expanded-1.0.0')
DATA_DIR = ROOT_DIRECTORY / 'sleep-cassette'
ARRAY_DIR = ROOT_DIRECTORY / 'cassette arrays - filtered'
CWT_OUT_DIR = ROOT_DIRECTORY / 'cassette cwt'
WSST_OUT_DIR = ROOT_DIRECTORY / 'cassette wsst'
TRIM_MINUTES = 30  # Keep 30 minutes of wake before/after sleep
WAKE_LABEL = 0
EPOCHS_TO_KEEP = TRIM_MINUTES * 2  # Each epoch is 30 seconds

all_filenames = sorted(os.listdir(DATA_DIR))
subject_ids = sorted(list(set([f.split('-')[0][:6] for f in all_filenames if f.endswith('edf')])))
print(f"Found {len(subject_ids)} subjects to process.")

CWT_OUT_DIR.mkdir(exist_ok=True)
WSST_OUT_DIR.mkdir(exist_ok=True)

all_cwt_paths = []
all_wsst_paths = []
all_final_labels = []

initial_total_epochs = 0
initial_class_counts = Counter()
initial_subject_stats = {}
final_subject_stats = {}

for subject_id in tqdm(subject_ids, desc="Processing Subjects"):
    signal_path = ARRAY_DIR / f'{subject_id}-signal.hdf5'
    label_path = ARRAY_DIR / f'{subject_id}-label.hdf5'

    if not signal_path.exists() or not label_path.exists():
        print(f"\nWarning: Data for subject {subject_id} not found. Skipping.")
        continue

    with h5py.File(signal_path, 'r') as f:
        x_subject = f['data'][:]
    with h5py.File(label_path, 'r') as f:
        y_subject = f['label'][:]

    y_subject[y_subject == 4] = 3
    y_subject[y_subject == 5] = 4

    initial_subject_stats[subject_id] = len(y_subject)
    initial_total_epochs += len(y_subject)
    initial_class_counts.update(y_subject)

    sleep_indices = np.where(y_subject != WAKE_LABEL)[0]
    if len(sleep_indices) == 0:
        print(f"\nInfo: Subject {subject_id} has no sleep stages. Discarding all epochs.")
        final_subject_stats[subject_id] = 0
        continue

    first_sleep_idx, last_sleep_idx = sleep_indices[0], sleep_indices[-1]
    start_idx = max(0, first_sleep_idx - EPOCHS_TO_KEEP)
    end_idx = min(len(y_subject), last_sleep_idx + EPOCHS_TO_KEEP + 1)

    x_subject_trimmed = x_subject[start_idx:end_idx]
    y_subject_trimmed = y_subject[start_idx:end_idx]
    final_subject_stats[subject_id] = len(y_subject_trimmed)

    subject_cwt_dir = CWT_OUT_DIR / subject_id
    subject_wsst_dir = WSST_OUT_DIR / subject_id
    subject_cwt_dir.mkdir(exist_ok=True)
    subject_wsst_dir.mkdir(exist_ok=True)

    for i in range(len(x_subject_trimmed)):
        signal_epoch = x_subject_trimmed[i]
        label_epoch = y_subject_trimmed[i]

        Tx, Wx, _, _ = ssq_cwt(np.swapaxes(signal_epoch, 0, 1), wavelet=Wavelet('morlet'), fs=100)
        cwt1 = cv2.resize(np.abs(Wx[0].cpu().numpy()), dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        cwt2 = cv2.resize(np.abs(Wx[1].cpu().numpy()), dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        cwt_stacked = np.stack([cwt1, cwt2], axis=0).astype(np.float16)

        wsst1 = cv2.resize(np.abs(Tx[0].cpu().numpy()), dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        wsst2 = cv2.resize(np.abs(Tx[1].cpu().numpy()), dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        wsst_stacked = np.stack([wsst1, wsst2], axis=0).astype(np.float16)

        cwt_filepath = subject_cwt_dir / f'{i:06d}.hdf5'
        wsst_filepath = subject_wsst_dir / f'{i:06d}.hdf5'

        with h5py.File(cwt_filepath, "w") as f:
            f.create_dataset("data", data=cwt_stacked, compression="gzip", compression_opts=9)

        with h5py.File(wsst_filepath, "w") as f:
            f.create_dataset("data", data=wsst_stacked, compression="gzip", compression_opts=9)

        all_cwt_paths.append(str(cwt_filepath))
        all_wsst_paths.append(str(wsst_filepath))
        all_final_labels.append(label_epoch)

print("\nSaving WSST and CWT file names and labels to CSV...")

cwt_df = pd.DataFrame({'file_path': all_cwt_paths, 'label': all_final_labels})
cwt_df.to_csv('./cwt.csv', index=False)

wsst_df = pd.DataFrame({'file_path': all_wsst_paths, 'label': all_final_labels})
wsst_df.to_csv('./wsst.csv', index=False)

print("\n" + "=" * 50)
print(" " * 15 + "PROCESSING REPORT")
print("=" * 50)

print("\n--- Per-Subject Epoch Counts ---")
print("{:<10} {:<10} {:<10}".format("Subject", "Initial", "Final"))
print("-" * 32)
for sid in subject_ids:
    if sid in initial_subject_stats:
        print("{:<10} {:<10} {:<10}".format(sid, initial_subject_stats.get(sid, 'N/A'),
                                            final_subject_stats.get(sid, 'N/A')))

print("\n--- Aggregate Report (Before vs. After Trimming) ---")
final_class_counts = Counter(all_final_labels)
final_total_epochs = len(all_final_labels)

print("\n1. Total Epochs:")
print(f"   - Before: {initial_total_epochs:,}")
print(f"   - After:  {final_total_epochs:,}")
epochs_removed = initial_total_epochs - final_total_epochs
print(f"   - Removed: {epochs_removed:,} epochs ({epochs_removed / initial_total_epochs:.2%})")

print("\n2. Class Distribution (Label: Count):")
all_labels = sorted(list(set(initial_class_counts.keys()) | set(final_class_counts.keys())))
print("   {:<10} {:<15} {:<15}".format("Label", "Before", "After"))
print("   " + "-" * 42)
for label in all_labels:
    before_count = initial_class_counts.get(label, 0)
    after_count = final_class_counts.get(label, 0)
    if label == WAKE_LABEL:
        stage_name = f"WAKE ({label})"
    elif label == 3:
        stage_name = f"N3+N4 ({label})"
    elif label == 4:
        stage_name = f"REM ({label})"
    else:
        stage_name = f"N{label} ({label})"

    print(f"   {stage_name:<10} {before_count:<15,} {after_count:<15,}")
    if label == WAKE_LABEL and before_count > 0:
        wake_removed = before_count - after_count
        print("   {:<10} {} {}".format("", "Removed:", f"{wake_removed:,} ({wake_removed / before_count:.2%})"))
print("   " + "-" * 42)
print("   {:<10} {:<15,} {:<15,}".format("TOTAL", initial_total_epochs, final_total_epochs))

print("\n" + "=" * 50)
print("Done!")