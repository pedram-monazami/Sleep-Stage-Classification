import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from .data.dataloader import get_dataloaders
from .models.factory import get_model
from .utils.logger import Logger
from .utils.early_stopping import EarlyStopping
from .utils.losses import FocalLoss
from .utils.metrics import calculate_metrics, compute_loss_per_class


class Trainer:
    """
    The main Trainer class that orchestrates the entire training and evaluation process.
    """

    def __init__(self, config, device, fold_num=None):
        self.config = config
        self.device = device
        self.fold_num = fold_num

        # Determine run name for logging
        self.run_name = f"cv_fold_{fold_num}" if fold_num is not None else "normal_train"
        self.output_dir = Path(config['output_dir']) / self.run_name
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize components
        self._setup_components()

    def _setup_components(self):
        """Initializes all components needed for training."""
        # --- Data Loaders ---
        self.train_loader, self.val_loader, _ = get_dataloaders(self.config, self.fold_num)

        # --- Model ---
        self.model = get_model(self.config).to(self.device)
        print(
            f"Model: {self.config['model']['name']}. Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # --- Optimizer ---
        optimizer_name = self.config['training']['optimizer']
        lr = self.config['training']['initial_lr']
        wd = self.config['training']['weight_decay']
        if optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:  # Default to SGD
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd)

        # --- Scheduler ---
        scheduler_name = self.config['training']['scheduler']
        if scheduler_name == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.config['training']['scheduler_factor'],
                patience=self.config['training']['scheduler_patience']
            )
        elif scheduler_name == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['step_size'],
                gamma=0.1
            )
        else:
            self.scheduler = None

        # --- Loss Function ---
        self.criterion = self._get_loss_function()

        # --- State Tracking ---
        self.start_epoch = 1
        self.best_metric = float('-inf') if self.config['early_stopping']['metric'] != 'val_loss' else float('inf')

        # --- Utilities ---
        self.logger = Logger(self.config, self.run_name)
        self.early_stopper = self._setup_early_stopping()

        # --- Resume from Checkpoint ---
        if self.config['model']['resume_from_checkpoint']:
            self._load_checkpoint(self.config['model']['resume_from_checkpoint'])

    def _get_loss_function(self, class_weights=None):
        """Creates the loss function based on config."""
        loss_name = self.config['training']['loss_function']
        if loss_name == 'FocalLoss':
            return FocalLoss(
                gamma=self.config['training']['focal_loss_gamma'],
                alpha=class_weights,
                task_type='multi-class',
                num_classes=self.config['model']['params']['num_classes']
            )
        else:  # Default to CrossEntropyLoss
            return nn.CrossEntropyLoss(weight=class_weights)

    def _setup_early_stopping(self):
        """Initializes the EarlyStopping utility."""
        if not self.config['early_stopping']['enabled']:
            return None

        metric = self.config['early_stopping']['metric']
        metric_type = "descending" if "loss" in metric else "ascending"

        return EarlyStopping(
            patience=self.config['early_stopping']['patience'],
            min_delta=self.config['early_stopping']['min_delta'],
            metric_type=metric_type
        )

    def _load_checkpoint(self, path):
        """Loads state from a checkpoint file."""
        print(f"Resuming training from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler and 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        if self.train_loader.sampler and 'sampler_state' in checkpoint:
            self.train_loader.sampler.load_state_dict(checkpoint['sampler_state'])
        if self.early_stopper and 'early_stopper_state' in checkpoint:
            self.early_stopper.load_state_dict(checkpoint['early_stopper_state'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        print(f"Resuming from end of epoch {checkpoint['epoch']}. Starting epoch {self.start_epoch}.")

    def _save_checkpoint(self, epoch, is_best):
        """Saves the current state to a checkpoint file."""
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_metric': self.best_metric
        }
        if self.scheduler:
            state['scheduler_state'] = self.scheduler.state_dict()
        if self.train_loader.sampler and hasattr(self.train_loader.sampler, 'state_dict'):
            state['sampler_state'] = self.train_loader.sampler.state_dict()
        if self.early_stopper:
            state['early_stopper_state'] = self.early_stopper.state_dict()

        if self.config['logging']['save_each_epoch']:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(state, epoch_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            print(f"Epoch {epoch}: New best model saved to {best_path}")

    def train(self):
        """The main training loop over all epochs."""
        for epoch in range(self.start_epoch, self.config['training']['epochs'] + 1):
            print(f"\n--- Epoch {epoch}/{self.config['training']['epochs']} ---")

            # --- Training Phase ---
            train_metrics = self._run_epoch(is_train=True)

            # --- Validation Phase ---
            val_metrics = self._run_epoch(is_train=False) if self.val_loader else None

            # --- Per-Class Loss Calculation (for logging or dynamic weighting) ---
            if self.config['logging']['save_loss_per_class_train']:
                train_metrics['class_losses'] = compute_loss_per_class(self.model, self.train_loader, self.device,
                                                                       self.config['model']['params']['num_classes'],
                                                                       self.config['data']['dataset_type'])

            if val_metrics and self.config['logging']['save_loss_per_class_val']:
                val_metrics['class_losses'] = compute_loss_per_class(self.model, self.val_loader, self.device,
                                                                     self.config['model']['params']['num_classes'],
                                                                     self.config['data']['dataset_type'])

            # --- Dynamic Class Weighting (updates criterion for the *next* epoch) ---
            if self.config['training']['dynamic_class_weighting']:
                base = self.config['training']['dynamic_class_weighting_base']
                print(f"Updating class weights dynamically based on '{base}' losses.")

                losses_for_weighting = None
                if base == 'val' and val_metrics:
                    # If we already calculated them for logging, use them. Otherwise, calculate now.
                    losses_for_weighting = (val_metrics.get('class_losses') or
                                            compute_loss_per_class(self.model, self.val_loader, self.device,
                                                                   self.config['model']['params']['num_classes'],
                                                                   self.config['data']['dataset_type']))
                elif base == 'train':
                    losses_for_weighting = (train_metrics.get('class_losses') or
                                            compute_loss_per_class(self.model, self.train_loader, self.device,
                                                                   self.config['model']['params']['num_classes'],
                                                                   self.config['data']['dataset_type']))

                if losses_for_weighting:
                    total_loss = sum(losses_for_weighting)
                    # Normalize losses to use as weights
                    weights = torch.tensor([l / total_loss for l in losses_for_weighting], device=self.device)
                    # Re-initialize the loss function with the new weights
                    self.criterion = self._get_loss_function(class_weights=weights)
                    print(f"New weights: {[round(w.item(), 4) for w in weights]}")

            # --- Logging ---
            self.logger.log_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=self.optimizer.param_groups[0]['lr']
            )

            # --- Scheduler & Early Stopping ---
            current_metric = -1
            if self.val_loader:
                metric_key = self.config['early_stopping']['metric'].replace('val_', '')
                current_metric = val_metrics[metric_key]
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()

                # Check for best model
                is_best = False
                if self.early_stopper.metric_type == "ascending" and current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
                elif self.early_stopper.metric_type == "descending" and current_metric < self.best_metric:
                    self.best_metric = current_metric
                    is_best = True

                self._save_checkpoint(epoch, is_best)

                if self.early_stopper:
                    self.early_stopper(current_metric)
                    if self.early_stopper.early_stop:
                        print(f"Early stopping triggered at epoch {epoch}.")
                        break
            else:
                # If no validation, save every epoch as 'best' is not defined
                self._save_checkpoint(epoch, is_best=False)

    def _run_epoch(self, is_train: bool):
        """
        Runs a single epoch of training or validation.
        """
        self.model.train(is_train)
        data_loader = self.train_loader if is_train else self.val_loader

        all_preds, all_labels = [], []
        total_loss = 0.0

        phase = "Train" if is_train else "Validation"
        progress_bar = tqdm(data_loader, desc=phase, leave=False)

        for data, labels in progress_bar:
            labels = labels.to(self.device)

            # Handle different input structures
            if self.config['data']['dataset_type'] in ["Dual", "SequencedDual"]:
                inputs = tuple(d.type(torch.FloatTensor).to(self.device, non_blocking=True) for d in data)
                outputs = self.model(*inputs)
            else:  # Single input
                inputs = data.type(torch.FloatTensor).to(self.device, non_blocking=True)
                outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(data_loader.dataset)
        metrics = calculate_metrics(all_labels, all_preds, self.config['model']['params']['num_classes'])
        metrics['loss'] = epoch_loss

        metrics['class_losses'] = None

        return metrics

    def evaluate(self, model, test_loader):
        """Runs final evaluation on the test set."""
        criterion = self._get_loss_function()  # Use standard criterion for eval
        test_metrics = self._run_epoch(model, test_loader, criterion, is_train=False)

        print("\n--- Test Set Evaluation Results ---")
        self.logger.log_final_evaluation(test_metrics)
