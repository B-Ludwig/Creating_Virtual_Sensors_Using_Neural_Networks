import os
import argparse
import json
import glob
import logging
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from utils.setupLogging import setup_logging
from dataModule.dataModule import TankDataModule
from utils.utils import create_dir
from model.net import LSTMModel
from model.opt import OptModuleBatch


def load_experiments(hparams_file: str) -> list:
    """
    Load experiments from a JSON configuration file.

    Args:
        hparams_file: Path to the JSON file containing an "experiments" list.

    Returns:
        A list of experiment hyperparameter dictionaries.
    """
    with open(hparams_file, 'r') as f:
        data = json.load(f)
    return data.get('experiments', [])


def main(experiment_folder: str):
    """
    Evaluate pretrained models on the test set, optimize missing sensor data,
    and summarize reconstruction losses.

    Args:
        experiment_folder: Directory containing best_hparams.json and model checkpoints.
    """
    setup_logging(logging.INFO)
    logging.info('Starting evaluation in %s', experiment_folder)

    hparams_path = os.path.join(experiment_folder, 'best_hparams.json')
    experiments = load_experiments(hparams_path) if os.path.isfile(hparams_path) else []

    # Prepare TensorBoard logger
    tb_dir = os.path.join(experiment_folder, 'tb_log')
    writer = SummaryWriter(tb_dir)

    all_loss_records = []  # Collect per-sample loss entries

    # Iterate through each experiment configuration
    for hparams in experiments:
        # Override batch size if needed
        hparams['TRAINING']['BATCH_SIZE'] = 2048
        logging.info('Loaded hyperparameters: %s', hparams)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info('Using device: %s', device)

        # Initialize data module and test loader
        data_module = TankDataModule(hparams)
        test_loader = data_module.get_test_dataloader()
        num_batches = len(test_loader)
        logging.info('Test loader batches: %d', num_batches)

        # Ensure plot output directory exists
        plot_dir = os.path.join(experiment_folder, 'plots_test')
        create_dir(experiment_folder, 'plots_test')

        # Locate checkpoint file
        ckpt_pattern = os.path.join(experiment_folder, 'model', '**', '.pth')
        checkpoint_paths = glob.glob(ckpt_pattern, recursive=True)

        # Process each checkpoint
        for ckpt_path in checkpoint_paths:
            logging.info('Evaluating checkpoint: %s', ckpt_path)

            # Build and load model
            model = LSTMModel(hparams=hparams).to(device)
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()

            input_dim = hparams['MODEL']['INPUT_DIM']

            # Optimize one sensor at a time
            for sensor_idx in range(input_dim):
                batch_counter = 0

                for batch in test_loader:
                    if batch_counter >= num_batches:
                        break

                    # Unpack batch tensors
                    x_gt = batch[0] if isinstance(batch, (list, tuple)) else batch
                    x_gt = x_gt.to(device).float()
                    y_gt = x_gt.clone()

                    # Prepare masked input and optimization mask
                    B, T, D = x_gt.shape
                    mask = torch.zeros_like(x_gt, dtype=torch.bool)
                    mask[:, :, sensor_idx] = True

                    if hparams['TRAINING']['MASKED']:
                        x_guess = x_gt.clone()

                        # One forward pass, since model was trained with mask                    
                        x_guess_new = x_gt.clone()
                        x_guess_new[:, :, sensor_idx] = -1
                        x_guess_new = x_guess_new.to(device).float()
                        model.init_weights(x_guess_new.size(0), gpu=(device.type == "cuda"))
                        x_guess_new = model(x_guess_new)

                        # Update x_guess without torch grad
                        with torch.no_grad():
                            x_guess[:, :, sensor_idx].copy_(x_guess_new[:, :, sensor_idx])
                        
                        y_guess = x_guess.clone()
                    else:
                        x_guess = x_gt.clone()
                        y_guess = y_gt.clone()
                        x_guess[:, :, sensor_idx] = 0
                        y_guess[:, :, sensor_idx] = 0

                    # Run optimization for missing sensor
                    optimizer = OptModuleBatch(hparams, model)
                    x_hat, y_hat, loss_vals, cycles = optimizer.find_params(
                        x_guess, y_guess, mask,
                        patience=250, max_consecutive_cycles=500
                    )

                    # Periodically save example plots
                    if batch_counter == 0:
                        plot_samples(
                            x_gt, y_hat, sensor_idx,
                            ckpt_path, batch_counter, plot_dir, hparams['DATA']['COLUMNS']
                        )

                    # Record losses for all samples
                    loss_per_sample = compute_losses(
                        model, x_gt, y_hat, y_gt, loss_vals, sensor_idx, device
                    )
                    all_loss_records.extend(loss_per_sample)

                    batch_counter += 1

    # Summarize and save results
    summarize_and_save(all_loss_records, experiment_folder)


def plot_samples(x_gt, y_hat, sensor_idx, ckpt_name, batch_idx, plot_dir, columns):
    """
    Plot reconstruction outputs for the first few samples of a batch.
    """
    y_mod = x_gt.clone().to(x_gt.device)
    y_mod[:, :, sensor_idx] = 0
    output = model(y_mod)

    for sample in range(min(10, x_gt.size(0))):
        sensor_name = columns[sensor_idx]
        plt.figure()
        plt.plot(x_gt[sample, :, sensor_idx].cpu(), label='Ground Truth')
        plt.plot(output[sample, :, sensor_idx].cpu(), ':', label='Model Recon')
        plt.plot(y_hat[sample, :, sensor_idx].cpu(), '--', label='Optimized Input')
        plt.title(sensor_name)
        plt.legend()
        filename = f"{os.path.basename(ckpt_name)}_batch{batch_idx}_sensor{sensor_idx}_sample{sample}.png"
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()


def compute_losses(model, x_gt, y_hat, y_gt, loss_vals, sensor_idx, device):
    """
    Calculate mean squared errors and package per-sample loss information.
    """
    model.eval()
    x_gt = x_gt.to(device)
    y_hat = y_hat.to(device)

    # Standard reconstruction for comparison
    y_std = model(x_gt)

    records = []
    for idx in range(x_gt.size(0)):
        mse_std = ((y_hat[idx, :, sensor_idx] - y_std[idx, :, sensor_idx])**2).mean().item()
        mse_gt = ((y_hat[idx, :, sensor_idx] - y_gt[idx, :, sensor_idx])**2).mean().item()
        records.append({
            'sensor': sensor_idx,
            'total_loss': loss_vals[idx].item(),
            'mse_vs_model': mse_std,
            'mse_vs_gt': mse_gt,
            'cycles': optimizer.cycles
        })
    return records


def summarize_and_save(records, output_dir):
    """
    Aggregate loss records by sensor and checkpoint, then save CSV summary.
    """
    df = pd.DataFrame(records)
    summary = df.groupby('sensor').agg([
        ('mean_total', 'mean'), ('std_total', 'std'),
        ('mean_mse_model', 'mean'), ('std_mse_model', 'std'),
        ('mean_mse_gt', 'mean'), ('std_mse_gt', 'std'),
        ('mean_cycles', 'mean'), ('std_cycles', 'std')
    ]).reset_index()

    csv_path = os.path.join(output_dir, 'loss_summary_reconstruction.csv')
    summary.to_csv(csv_path, index=False)
    logging.info('Saved loss summary: %s', csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optimize missing sensor data and summarize reconstruction losses.'
    )
    parser.add_argument(
        '--experiment_folder', required=True,
        help='Directory with best_hparams.json and model checkpoints.'
    )
    args = parser.parse_args()
    main(args.experiment_folder)
