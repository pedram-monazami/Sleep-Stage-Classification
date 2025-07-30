import pandas as pd
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np


class Logger:
    def __init__(self, config, run_name):
        self.config = config
        self.output_dir = Path(config['output_dir']) / run_name
        self.metrics_file = self.output_dir / f'{run_name}_metrics.csv'
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        self.class_names = config['model']['class_names']

    def _plot_cm(self, cm, title, normalize=False):
        """Generates a plottable image of a confusion matrix."""
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
            fmt = ".2f"
        else:
            fmt = "d"

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        image = ToTensor()(Image.open(buf).convert("RGB"))
        return image

    def _compute_per_class_metrics(self, cm):
        """Computes precision, recall, and F1 for each class from a confusion matrix."""
        num_classes = len(self.class_names)
        per_class_metrics = {'precision': [], 'recall': [], 'f1': []}
        for i in range(num_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class_metrics['precision'].append(precision)
            per_class_metrics['recall'].append(recall)
            per_class_metrics['f1'].append(f1)
        return per_class_metrics

    def log_epoch(self, epoch, train_metrics, val_metrics, lr):
        """Logs metrics for one epoch to CSV and TensorBoard."""
        row = {'epoch': epoch, 'lr': lr}

        train_per_class = self._compute_per_class_metrics(train_metrics['cm'])
        train_metrics['f1_per_class'] = train_per_class['f1']
        train_metrics['precision_per_class'] = train_per_class['precision']
        train_metrics['recall_per_class'] = train_per_class['recall']

        if val_metrics:
            val_per_class = self._compute_per_class_metrics(val_metrics['cm'])
            val_metrics['f1_per_class'] = val_per_class['f1']
            val_metrics['precision_per_class'] = val_per_class['precision']
            val_metrics['recall_per_class'] = val_per_class['recall']

        def add_metrics_to_row(metrics, prefix):
            for key, value in metrics.items():
                if key == 'cm':
                    row[f'{prefix}_{key}'] = json.dumps(value.tolist())
                elif key.endswith('_per_class'):
                    metric_name = key.replace('_per_class', '')
                    for i, v in enumerate(value):
                        row[f'{prefix}_{self.class_names[i]}_{metric_name}'] = v
                elif key == 'class_losses' and isinstance(value, list):
                    for i, v in enumerate(value):
                        row[f'{prefix}_{self.class_names[i]}_{key.replace("class_", "")}'] = v  # e.g., train_Wake_loss
                else:
                    row[f'{prefix}_{key}'] = value

        add_metrics_to_row(train_metrics, 'train')
        if val_metrics:
            add_metrics_to_row(val_metrics, 'val')

        df = pd.read_csv(self.metrics_file) if self.metrics_file.exists() else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.metrics_file, index=False)

        # --- Comprehensive TensorBoard Logging ---

        def log_metrics_to_tb(metrics_dict, prefix, epoch_num):
            """Helper function to log a set of metrics to TensorBoard."""
            self.writer.add_scalar(f'Loss/{prefix}', metrics_dict['loss'], epoch_num)
            self.writer.add_scalar(f'Accuracy/{prefix}', metrics_dict['accuracy'], epoch_num)
            self.writer.add_scalar(f'F1_Macro/{prefix}', metrics_dict['f1'], epoch_num)
            self.writer.add_scalar(f'Precision_Macro/{prefix}', metrics_dict['precision'], epoch_num)
            self.writer.add_scalar(f'Recall_Macro/{prefix}', metrics_dict['recall'], epoch_num)
            self.writer.add_scalar(f'Kappa/{prefix}', metrics_dict['kappa'], epoch_num)

            for i, class_name in enumerate(self.class_names):
                self.writer.add_scalar(f'PerClass_F1/{prefix}_{class_name}', metrics_dict['f1_per_class'][i], epoch_num)
                self.writer.add_scalar(f'PerClass_Precision/{prefix}_{class_name}', metrics_dict['precision_per_class'][i], epoch_num)
                self.writer.add_scalar(f'PerClass_Recall/{prefix}_{class_name}', metrics_dict['recall_per_class'][i], epoch_num)

            if metrics_dict.get('class_losses') is not None:
                for i, class_name in enumerate(self.class_names):
                    self.writer.add_scalar(f'PerClass_Loss/{prefix}_{class_name}', metrics_dict['class_losses'][i], epoch_num)

            cm_image = self._plot_cm(metrics_dict['cm'], f'{prefix.title()} Confusion Matrix')
            cm_image_norm = self._plot_cm(metrics_dict['cm'], f'{prefix.title()} Confusion Matrix (Normalized)', normalize=True)
            self.writer.add_image(f'ConfusionMatrix/{prefix}', cm_image, epoch_num)
            self.writer.add_image(f'ConfusionMatrix_Normalized/{prefix}', cm_image_norm, epoch_num)

        # Log training metrics
        log_metrics_to_tb(train_metrics, "Train", epoch)

        # Log validation metrics if they exist
        if val_metrics:
            log_metrics_to_tb(val_metrics, "Validation", epoch)

        self.writer.add_scalar('LearningRate', lr, epoch)
        self.writer.flush()

        # --- Console Logging ---
        verbose = self.config['logging']['verbose']
        if verbose > 1:
            train_log = f"Train -> Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}"
            print(train_log)
            if val_metrics:
                val_log = f"Val   -> Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}"
                print(val_log)

    def log_final_evaluation(self, metrics):
        """Prints the final evaluation results to the console."""
        print("\n--- Test Set Evaluation Results ---")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (Macro): {metrics['f1']:.4f}")
        print(f"  Precision (Macro): {metrics['precision']:.4f}")
        print(f"  Recall (Macro): {metrics['recall']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['cm'])