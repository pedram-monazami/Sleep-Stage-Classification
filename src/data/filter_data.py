import numpy as np
import pyedflib
import os
from tqdm import tqdm
from pathlib import Path
import h5py
from scipy.signal import butter, lfilter


ROOT_DIRECTORY = Path('./data/sleep-edf-database-expanded-1.0.0')
SOURCE_DATA_DIR = ROOT_DIRECTORY / 'sleep-cassette'
OUTPUT_DIR = ROOT_DIRECTORY / 'cassette arrays - filtered'
LOWCUT = 0.1
HIGHCUT = 30.0
FS = 100
EPOCH_DURATION_SAMPLES = 3000  # 30 seconds * 100 Hz

LABELS_TO_REMOVE = [6, 7]  # '?' and 'Movement time'


def butter_bandpass(lowcut, highcut, fs, order=4):
    """Designs a Butterworth bandpass filter."""
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Applies a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def convert_label(label_string: str) -> int:
    """Maps the annotation string to an integer label."""
    label_map = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 4,
        'Sleep stage R': 5,
        'Sleep stage ?': 6,
        'Movement time': 7,
    }
    return label_map.get(label_string, 6)


def find_subject_files(subject_id: str, directory: Path) -> (Path, Path):
    """Finds the PSG and Hypnogram EDF files for a given subject ID."""
    try:
        psg_file = next(directory.glob(f"{subject_id}*-PSG.edf"))
        hypno_file = next(directory.glob(f"{subject_id}*-Hypnogram.edf"))
        return psg_file, hypno_file
    except StopIteration:
        return None, None


OUTPUT_DIR.mkdir(exist_ok=True)
all_files = os.listdir(SOURCE_DATA_DIR)
subject_ids = sorted(list(set([f.split('-')[0][:6] for f in all_files if f.endswith('.edf')])))
print(f"Found {len(subject_ids)} subjects to process.")

for subject_id in tqdm(subject_ids, desc="Processing Subjects"):
    psg_path, hypno_path = find_subject_files(subject_id, SOURCE_DATA_DIR)

    if not psg_path or not hypno_path:
        print(f"\nWarning: Missing PSG or Hypnogram file for subject {subject_id}. Skipping.")
        continue

    psg_file = pyedflib.EdfReader(str(psg_path))

    signal_1 = psg_file.readSignal(0)
    signal_2 = psg_file.readSignal(1)
    psg_file.close()

    # Apply the bandpass filter to each channel
    filtered_1 = butter_bandpass_filter(signal_1, LOWCUT, HIGHCUT, FS)
    filtered_2 = butter_bandpass_filter(signal_2, LOWCUT, HIGHCUT, FS)

    # Stack signals into a (n_samples, 2) array
    signals = np.stack([filtered_1, filtered_2], axis=1).astype(np.float32)

    hypno_file = pyedflib.EdfReader(str(hypno_path))
    annotations = hypno_file.readAnnotations()
    hypno_file.close()

    onsets, durations, labels_str = annotations

    # Use efficient Python lists to collect epochs and labels
    epoch_list = []
    label_list = []
    for i in range(len(labels_str)):
        onset_samples = int(onsets[i] * FS)
        duration_samples = int(durations[i] * FS)

        # Get the signal segment for the full annotation
        signal_segment = signals[onset_samples: onset_samples + duration_samples]

        # Reshape into epochs of 30 seconds
        segmented_epochs = signal_segment.reshape(-1, EPOCH_DURATION_SAMPLES, 2)

        if segmented_epochs.shape[0] == 0:
            continue

        num_epochs = segmented_epochs.shape[0]

        # Create corresponding labels
        label = convert_label(labels_str[i])
        segmented_labels = np.full(num_epochs, label, dtype=np.int8)

        epoch_list.append(segmented_epochs)
        label_list.append(segmented_labels)

    if not epoch_list:
        print(f"\nWarning: No valid epochs found for subject {subject_id}. Skipping.")
        continue

    X_subject = np.concatenate(epoch_list, axis=0)
    Y_subject = np.concatenate(label_list, axis=0)

    mask = ~np.isin(Y_subject, LABELS_TO_REMOVE)
    X_subject_clean = X_subject[mask]
    Y_subject_clean = Y_subject[mask]

    if X_subject_clean.shape[0] > 0:
        with h5py.File(OUTPUT_DIR / f'{subject_id}-signal.hdf5', "w") as f:
            f.create_dataset("data", data=X_subject_clean, compression="gzip", compression_opts=9)
        with h5py.File(OUTPUT_DIR / f'{subject_id}-label.hdf5', "w") as f:
            f.create_dataset("label", data=Y_subject_clean, compression="gzip", compression_opts=9)

print("\nDone!")