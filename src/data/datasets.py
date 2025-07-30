import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from collections import defaultdict


class DualSleepDataset(Dataset):
    def __init__(self, cwt_file_paths, wsst_file_paths, labels, cwt_transform=None,  wsst_transform=None):
        self.cwt_file_paths = cwt_file_paths
        self.wsst_file_paths = wsst_file_paths
        self.labels = labels
        self.cwt_transform = cwt_transform
        self.wsst_transform = wsst_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        with h5py.File(self.cwt_file_paths[idx], 'r') as f:
            cwt_data = f['data'][:]
        with h5py.File(self.wsst_file_paths[idx], 'r') as f:
            wsst_data = f['data'][:]
        if self.cwt_transform:
            cwt_data = self.cwt_transform(np.swapaxes(cwt_data, 0, 2))
        if self.wsst_transform:
            wsst_data = self.wsst_transform(np.swapaxes(wsst_data, 0, 2))

        return (cwt_data, wsst_data), self.labels[idx]


class DualConcatSleepDataset(Dataset):
    def __init__(self, cwt_file_paths, wsst_file_paths, labels, cwt_transform=None,  wsst_transform=None):
        self.cwt_file_paths = cwt_file_paths
        self.wsst_file_paths = wsst_file_paths
        self.labels = labels
        self.cwt_transform = cwt_transform
        self.wsst_transform = wsst_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        with h5py.File(self.cwt_file_paths[idx], 'r') as f:
            cwt_data = f['data'][:]
        with h5py.File(self.wsst_file_paths[idx], 'r') as f:
            wsst_data = f['data'][:]
        if self.cwt_transform:
            cwt_data = self.cwt_transform(np.swapaxes(cwt_data, 0, 2))
        if self.wsst_transform:
            wsst_data = self.wsst_transform(np.swapaxes(wsst_data, 0, 2))

        return torch.concat([cwt_data, wsst_data], dim=0), self.labels[idx]


class SingleSleepDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        with h5py.File(self.file_paths[idx], 'r') as f:
            data = f['data'][:]
        if self.transform:
            data = self.transform(np.swapaxes(data, 0, 2))

        return data, self.labels[idx]


class SequencedDualSleepDataset(Dataset):
    def __init__(self, cwt_file_paths, wsst_file_paths, labels, cwt_transform=None, wsst_transform=None, pool_priority=[3, 1, 4, 0, 2], seq_length=5):
        self.cwt_file_paths = cwt_file_paths
        self.wsst_file_paths = wsst_file_paths
        self.labels = labels
        self.cwt_transform = cwt_transform
        self.wsst_transform = wsst_transform
        self.seq_length = seq_length

        assert len(cwt_file_paths) == len(wsst_file_paths) == len(labels)

        self.pool_priority = pool_priority
        self.sequences_by_class = defaultdict(list)
        self._build_sequences()

        # Flatten all class pools into a single list for dataset access
        self.sequences = []
        for label, seqs in self.sequences_by_class.items():
            self.sequences.extend(seqs)

    def _get_pool_label(self, label_seq, strategy="majority"):
        if strategy == "majority":
            return max(set(label_seq), key=label_seq.count)
        elif strategy == "prioritized":
            for priority_label in self.pool_priority:
                if priority_label in label_seq:
                    return priority_label
        raise ValueError("Unknown pooling strategy: {}".format(strategy))

    def _build_sequences(self):
        for i in range(len(self.labels) - self.seq_length + 1):
            cwt_seq = self.cwt_file_paths[i:i + self.seq_length]
            wsst_seq = self.wsst_file_paths[i:i + self.seq_length]
            label_seq = self.labels[i:i + self.seq_length]
            pool_label = self._get_pool_label(list(label_seq), strategy="majority")
            self.sequences_by_class[pool_label].append((cwt_seq, wsst_seq, label_seq, pool_label))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        cwt_seq_paths, wsst_seq_paths, label_seq, pool_label = self.sequences[idx]

        cwt_data = []
        wsst_data = []
        for cwt_path, wsst_path in zip(cwt_seq_paths, wsst_seq_paths):
            with h5py.File(cwt_path, 'r') as f:
                x_cwt = f['data'][:]
            with h5py.File(wsst_path, 'r') as f:
                x_wsst = f['data'][:]

            if self.cwt_transform:
                x_cwt = self.cwt_transform(np.swapaxes(x_cwt, 0, 2))
            if self.wsst_transform:
                x_wsst = self.wsst_transform(np.swapaxes(x_wsst, 0, 2))

            cwt_data.append(x_cwt)
            wsst_data.append(x_wsst)

        cwt_data = torch.stack(cwt_data)  # Shape: (5, C, H, W)
        wsst_data = torch.stack(wsst_data)
        label_seq = torch.tensor(label_seq)
        return (cwt_data, wsst_data), label_seq
