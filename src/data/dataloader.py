import random
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from .datasets import SingleSleepDataset, DualSleepDataset, DualConcatSleepDataset, SequencedDualSleepDataset
from .sampler import BalancedSubsetSampler


def seed_worker(worker_id):
    """
    Sets the seed for a DataLoader worker.
    Ensures that random operations using `random` and `numpy` are reproducible
    across runs and workers.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def filter_by_subject(df, subjects):
    """Filters a dataframe to include only rows related to a list of subject folders."""
    if not subjects:
        return pd.DataFrame(columns=df.columns)
    return df[df['file_path'].apply(lambda path: any(subj in Path(path).parent.name for subj in subjects))].reset_index(
        drop=True)


def get_dataloaders(config, fold_num=None, is_eval=False):
    """
    The main factory function for creating data loaders.

    Args:
        config (dict): The main configuration dictionary.
        fold_num (int, optional): The 1-indexed fold number for cross-validation.
        is_eval (bool): If True, creates only the test loader for final evaluation.

    Returns:
        tuple: A tuple of (train_loader, val_loader, test_loader). Some may be None.
    """
    data_cfg = config['data']
    cwt_df = pd.read_csv(os.path.join(config['data_dir'], 'cwt.csv'))
    wsst_df = pd.read_csv(os.path.join(config['data_dir'], 'wsst.csv'))

    train_subjects, val_subjects, test_subjects = [], [], []

    if is_eval:
        # Evaluation mode: only need the test set
        test_split_df = pd.read_csv(data_cfg['test_subjects_csv'])
        test_subjects = test_split_df['subject_folder'].tolist()
    elif config['cross_validation']['enabled']:
        # Cross-validation mode
        folds_data = pd.read_csv(data_cfg['subject_split_csv'])
        fold_subjects = folds_data[folds_data['fold'] == fold_num]
        train_subjects = fold_subjects[fold_subjects['split'] == 'train']['subject_folder'].tolist()
        val_subjects = fold_subjects[fold_subjects['split'] == 'val']['subject_folder'].tolist()
    else:
        # Fixed split mode
        split_data = pd.read_csv(data_cfg['subject_split_csv'])
        train_subjects = split_data[split_data['split'] == 'train']['subject_folder'].tolist()
        val_subjects = split_data[split_data['split'] == 'val']['subject_folder'].tolist()
        if data_cfg.get('test_subjects_csv'):
            test_split_df = pd.read_csv(data_cfg['test_subjects_csv'])
            test_subjects = test_split_df['subject_folder'].tolist()

    # --- Create Datasets ---
    def create_dataset(subjects):
        if not subjects:
            return None

        cwt_subset_df = filter_by_subject(cwt_df, subjects)
        wsst_subset_df = filter_by_subject(wsst_df, subjects)

        # Determine normalization file paths
        norm_suffix = f"_fold{fold_num}.npy" if config['cross_validation']['enabled'] else ".npy"
        norm_dir = config['normalization_stats_dir']

        cwt_mean = np.load(os.path.join(norm_dir, f"cwt_mean{norm_suffix}"))
        cwt_std = np.load(os.path.join(norm_dir, f"cwt_std{norm_suffix}"))
        wsst_mean = np.load(os.path.join(norm_dir, f"wsst_mean{norm_suffix}"))
        wsst_std = np.load(os.path.join(norm_dir, f"wsst_std{norm_suffix}"))

        cwt_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cwt_mean, std=cwt_std)])
        wsst_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=wsst_mean, std=wsst_std)])

        dataset_type = data_cfg['dataset_type']
        dataset_args = {
            "cwt_file_paths": cwt_subset_df['file_path'].values,
            "wsst_file_paths": wsst_subset_df['file_path'].values,
            "labels": cwt_subset_df['label'].values,
            "cwt_transform": cwt_transform,
            "wsst_transform": wsst_transform
        }

        if dataset_type == "Single":
            return SingleSleepDataset(file_paths=cwt_subset_df['file_path'].values,
                                      labels=cwt_subset_df['label'].values, transform=cwt_transform)
        elif dataset_type == "Dual":
            return DualSleepDataset(**dataset_args)
        elif dataset_type == "DualConcat":
            return DualConcatSleepDataset(**dataset_args)
        elif dataset_type == "SequencedDual":
            return SequencedDualSleepDataset(**dataset_args, seq_length=5)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

    train_dataset = create_dataset(train_subjects)
    val_dataset = create_dataset(val_subjects)
    test_dataset = create_dataset(test_subjects)

    if is_eval:
        test_loader = DataLoader(
            test_dataset,
            batch_size=data_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg['num_workers'],
            worker_init_fn=seed_worker
        ) if test_dataset else None
        return None, None, test_loader

    # --- Create DataLoaders ---
    g = torch.Generator()
    g.manual_seed(config['seed'])

    sampler = None
    shuffle = True
    if data_cfg['balanced_subset_training'] and train_dataset:
        class_counts = np.bincount(train_dataset.labels)
        samples = min(class_counts) if data_cfg['samples_per_class'] == "min_class" else data_cfg['samples_per_class']
        sampler = BalancedSubsetSampler(train_dataset, samples_per_class=samples)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_cfg['num_workers'],
        generator=g,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=True if data_cfg['num_workers'] > 0 else False
    ) if train_dataset else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=True if data_cfg['num_workers'] > 0 else False
    ) if val_dataset else None

    return train_loader, val_loader, None
