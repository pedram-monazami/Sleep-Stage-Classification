# Modular Sleep Stage Classification Framework

This repository contains a robust and configurable framework for training and evaluating deep learning models for sleep stage classification. It is designed to work with pre-processed CWT (Continuous Wavelet Transform) and WSST (Wavelet Synchrosqueezed Transform) images derived from sleep EDF data. The framework is built with PyTorch and is designed to be modular, reusable, and easily extensible for various models and research experiments.

## Features

-   **Configuration-Driven:** All aspects of an experiment (paths, model, hyperparameters, etc.) are controlled via a single `config.yaml` file.
-   **Command-Line Overrides:** Easily override any configuration parameter directly from the command line for quick experiments.
-   **Cross-Validation & Fixed Splits:** Natively supports k-fold cross-validation as well as standard train/validation/test splits.
-   **Complete Resumability:** Save and load the complete training state (model, optimizer, scheduler, sampler, early stopping state, and epoch number) to seamlessly resume interrupted runs.
-   **Decoupled Modes:** A clear distinction between `train` mode for training models and `eval` mode for final testing on a hold-out set.
-   **Model & Dataset Factories:** Easily add new models or dataset types without changing the core training logic. Just add your new class and register it in the corresponding factory file.
-   **Comprehensive Logging:** Automatic logging of metrics, confusion matrices, and hyperparameters to both CSV files and TensorBoard for easy monitoring and comparison.
-   **Modular Structure:** A clean, organized codebase that separates concerns (data handling, models, training logic, utilities) for better maintainability and scalability.

## Project Structure

The project is organized into a `src` directory to keep the codebase clean and importable.

```
sleep_classification/
├── config.yaml               # Main configuration file for experiments
├── main.py                   # The main entry point to run the framework
├── README.md                 # This documentation file
└── src/
    ├── data/
    │   ├── data_loader.py      # Factory for creating datasets and dataloaders
    │   ├── datasets.py         # All torch.utils.data.Dataset class definitions
    │   └── sampler.py          # Custom samplers (e.g., BalancedSubsetSampler)
    ├── models/
    │   ├── factory.py          # Model factory to build models from config
    │   └── vision_transformer.py # Example file for model architecture code
    ├── utils/
    │   ├── early_stopping.py   # EarlyStopping utility class
    │   ├── logger.py           # Handles all logging (CSV, TensorBoard, console)
    │   ├── losses.py           # Custom loss functions (e.g., FocalLoss)
    │   └── metrics.py          # Helper functions for calculating metrics
    └── trainer.py              # The core Trainer class that manages the training loop
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd sleep_classification
    ```

2.  **Create a Python virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file with the following content and then install it.
    ```
    torch
    torchvision
    numpy
    pandas
    scikit-learn
    pyyaml
    tqdm
    seaborn
    matplotlib
    h5py
    tensorboard
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

The framework expects data to be organized as follows:

1.  **Image Data:** The CWT/WSST images should be stored as `.h5` files.
2.  **Metadata CSVs:** In your `data_dir` (specified in `config.yaml`), you must have:
    -   `cwt.csv`: A CSV file with columns `file_path` and `label`. Each row points to a CWT image file and its corresponding sleep stage label.
    -   `wsst.csv`: Same as above, but for WSST images. The order of epochs must match `cwt.csv`.
3.  **Subject Split Files:** These CSV files define which subjects belong to which data split (train, validation, test). Each file must have `subject_folder` and `split` columns. For cross-validation, an additional `fold` column (1-indexed) is required.
4.  **Normalization Statistics:** Pre-computed channel-wise mean and standard deviation values must be saved as `.npy` files in the `normalization_stats_dir`.
    -   For fixed splits: `cwt_mean.npy`, `cwt_std.npy`, etc.
    -   For cross-validation: `cwt_mean_fold1.npy`, `cwt_std_fold1.npy`, etc.

## How to Run

The framework is controlled via the `main.py` script and the `config.yaml` file.

### 1. Configure Your Experiment

Modify `config.yaml` to define your experiment. Pay close attention to the `data`, `model`, and `cross_validation` sections.

### 2. Run from the Command Line

#### **Start a New Cross-Validation Run**
This will run all 10 folds as specified in the default config.

```bash
python main.py --config config.yaml --mode train
```

#### **Run a Single, Specific Fold**
To run only the 3rd fold of a cross-validation setup:

```bash
python main.py --config config.yaml --mode train --cross_validation.run_fold 3
```

#### **Start a Standard (Non-CV) Training Run**
First, set `cross_validation.enabled: false` in `config.yaml` and ensure your `subject_split_csv` points to a file with 'train' and 'val' splits.

```bash
python main.py --config config.yaml --mode train
```

#### **Resume a Stopped Training Run**
If a run was interrupted, find the latest checkpoint file (e.g., `outputs/SingleViT/cv_fold_1/checkpoints/epoch_12.pt`). Update your config to point to it and re-run the *same* command you used initially.

In `config.yaml`:
```yaml
model:
  resume_from_checkpoint: outputs/SingleViT/cv_fold_1/checkpoints/epoch_12.pt```
Then run (for example, to resume fold 1):
```
```bash
python main.py --config config.yaml --mode train --cross_validation.run_fold 1
```
The trainer will load all states and continue from epoch 13.

#### **Evaluate a Trained Model**
To evaluate your best-trained model on the test set, set the mode to `eval` and provide the path to the model checkpoint.

In `config.yaml`:
```yaml
mode: eval
model:
  resume_from_checkpoint: outputs/SingleViT/cv_fold_1/checkpoints/best_model.pt
```
Then run:
```bash
python main.py --config config.yaml
```

## Extending the Framework

The framework is designed to be easily extended.

### Adding a New Model

1.  **Create the Model Class:** Add your model's architecture in a Python file within `src/models/`, for instance, `src/models/my_new_cnn.py`.
2.  **Register the Model:** Open `src/models/factory.py`.
    -   Import your new model class: `from .my_new_cnn import MyCNN`
    -   Add an entry to the `models` dictionary: `'MyCNN': MyCNN`
3.  **Use in Config:** You can now use your new model by setting `model.name: MyCNN` in `config.yaml`.

### Adding a New Dataset Type

1.  **Create the Dataset Class:** Define your new `Dataset` class in `src/data/datasets.py`.
2.  **Integrate into the Factory:** Open `src/data/data_loader.py`. In the `create_dataset` function, add an `elif` condition for your new `dataset_type` name and return an instance of your new class.
3.  **Use in Config:** You can now use your new dataset by setting `data.dataset_type: "YourNewType"` in `config.yaml`.