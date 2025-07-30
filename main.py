import argparse
import yaml
import os
import random
import numpy as np
import torch
from src.trainer import Trainer
from src.data.dataloader import get_dataloaders
from src.models.factory import get_model


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config_device):
    """Gets the torch device."""
    if config_device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(config_device)


def main(cfg):
    """Main function to run training or evaluation."""
    set_seed(cfg['seed'])
    device = get_device(cfg['device'])
    print(f"Using device: {device}")

    if cfg['mode'] == 'train':
        if cfg['cross_validation']['enabled']:
            start_fold = 1
            end_fold = cfg['cross_validation']['num_folds']

            if cfg['cross_validation']['run_fold'] != -1:
                start_fold = cfg['cross_validation']['run_fold']
                end_fold = cfg['cross_validation']['run_fold']

            for fold in range(start_fold, end_fold + 1):
                print(f"\n{'=' * 20} FOLD {fold}/{cfg['cross_validation']['num_folds']} {'=' * 20}")
                trainer = Trainer(cfg, device, fold_num=fold)
                trainer.train()
                print(f"{'=' * 20} FOLD {fold} COMPLETE {'=' * 20}")
        else:
            # Normal training with a fixed split
            print(f"\n{'=' * 20} STARTING NORMAL TRAINING {'=' * 20}")
            trainer = Trainer(cfg, device, fold_num=None)
            trainer.train()
            print(f"{'=' * 20} NORMAL TRAINING COMPLETE {'=' * 20}")

    elif cfg['mode'] == 'eval':
        print(f"\n{'=' * 20} STARTING EVALUATION MODE {'=' * 20}")
        if not cfg['model']['resume_from_checkpoint'] or not os.path.exists(cfg['model']['resume_from_checkpoint']):
            raise ValueError("For 'eval' mode, 'model.resume_from_checkpoint' must be a valid path to a model file.")

        _, _, test_loader = get_dataloaders(cfg, fold_num=None, is_eval=True)
        if not test_loader:
            raise ValueError("Test dataloader could not be created. Ensure 'test_subjects_csv' is set in the config.")

        model = get_model(cfg).to(device)

        # Load the state dict for evaluation
        checkpoint = torch.load(cfg['model']['resume_from_checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state'])

        eval_trainer = Trainer(cfg, device, fold_num=None)  # Trainer is used for its utility functions
        eval_trainer.evaluate(model, test_loader)

        print(f"{'=' * 20} EVALUATION COMPLETE {'=' * 20}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sleep Stage Classification Training Framework")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')

    args, unknown = parser.parse_known_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)


    # Function to recursively update dict
    def update_config(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_config(d.get(k, {}), v)
            else:
                d[k] = v
        return d


    # Parse unknown args for overrides
    for i in range(0, len(unknown), 2):
        key_str = unknown[i].replace('--', '')
        val = unknown[i + 1]

        try:
            val = eval(val)
        except (NameError, SyntaxError):
            pass

        keys = key_str.split('.')
        d = cfg
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = val

    print("--- Running with Configuration ---")
    print(yaml.dump(cfg, default_flow_style=False))
    print("---------------------------------")

    main(cfg)
