import os
import argparse
import json
import numpy as np
import random
import torch
import logging
from torch.utils.tensorboard import SummaryWriter

from utils.setupLogging import setup_logging
from dataModule.dataModule import TankDataModule
from utils.utils import create_dir
from model.train import TrainModule


def load_experiments(hparams_file):
    """
    Load experiment configurations from a JSON file.

    Args:
        hparams_file (str): Path to the JSON file containing experiments.

    Returns:
        list of dict: A list of hyperparameter dictionaries, or an empty list.
    """
    with open(hparams_file, 'r') as f:
        data = json.load(f)
    return data.get("experiments", [])


def main(experiment_folder):
    """
    Configure and run training for each experiment defined in the hparams file
    or in the default settings.
    
    Args:
        experiment_folder (str): Directory where hparams.json may reside and
                                 where experiment outputs will be written.
    """
    # Initialize logging at INFO level (use DEBUG for more verbosity)
    setup_logging(logging.INFO)

    print(f"Running experiments in: {experiment_folder}")

    # Locate the hyperparameter file
    hparams_file = os.path.join(experiment_folder, "hparams.json")

    if os.path.isfile(hparams_file):
        # Load experiments from JSON
        experiments = load_experiments(hparams_file)
    else:
        # Fallback to a single default experiment configuration
        experiments = [
            {
                "MODEL": {
                    "DIR": "model",
                    "NAME": "LSTMModel",
                    "INPUT_DIM": 39,
                    "HIDDEN_DIM": 117,
                    "N_LAYERS": 2,
                    "ENCODER_DIM": 26,
                    "OUTPUT_DIM": 39,
                    "DROPOUT": 0
                },
                "TRAINING": {
                    "DIR": "train",
                    "BATCH_SIZE": 512,
                    "MAX_EPOCHS": 3000,
                    "LR": 0.001,
                    "WEIGHT_DECAY": 0,
                    "NOISE": 0.01,
                    "MASKED": "FALSE"
                },
                "DATA": {
                    # Insert correct path to ds6_hybrid_s
                    "DIR": "data",
                    "COLUMNS": [
                        "bottling0.tank_B402.level",
                        "bottling0.sensor_continuous_pressure_tank_B402.p",
                        "bottling1.tank_B402.level",
                        "bottling1.sensor_continuous_pressure_tank_B402.p",
                        "bottling0.pump_n_in",
                        "bottling0.pump_P401.N_in",
                        "bottling0.sensor_continuous_pressure_pump_P401.p",
                        "bottling0.sensor_continuous_volumeFlowRate.V_flow",
                        "bottling0.sensor_continuous_volumeFlowRate.port_a.m_flow",
                        "bottling0.sensor_continuous_volumeFlowRate.port_b.m_flow",
                        "bottling1.pump_n_in",
                        "bottling1.pump_P401.N_in",
                        "bottling1.sensor_continuous_pressure_pump_P401.p",
                        "bottling1.sensor_continuous_volumeFlowRate.V_flow",
                        "bottling1.sensor_continuous_volumeFlowRate.port_a.m_flow",
                        "bottling1.sensor_continuous_volumeFlowRate.port_b.m_flow",
                        "distill0.sensor_continuous_pressure_tank_B101.p",
                        "distill0.pump_P101.N_in",
                        "distill0.pump_n_in",
                        "distill0.sensor_continuous_volumeFlowRate.port_b.m_flow",
                        "distill0.sensor_continuous_volumeFlowRate.V_flow",
                        "distill0.sensor_continuous_volumeFlowRate.port_a.m_flow",
                        "distill0.tank_B103.level",
                        "distill0.sensor_continuous_pressure_tank_B103.p",
                        "distill0.tank_B101.level",
                        "distill0.sensor_continuous_pressure_distill_out1.p",
                        "distill0.heater_distill.port.Q_flow",
                        "distill0.distill.level",
                        "distill0.sensor_continuous_pressure_distill_out0.p",
                        "bottling0.tank_B401.level",
                        "bottling0.sensor_continuous_pressure_tank_B401.p",
                        "distill0.tank_B102.level",
                        "distill0.sensor_continuous_pressure_tank_B102.p",
                        "bottling1.tank_B401.level",
                        "bottling1.sensor_continuous_pressure_tank_B401.p",
                        "distill0.sensor_continuous_pressure_pump_P101.p",
                        "distill0.heater_distill.port.T",
                        "distill0.cooler_B102.port.T",
                        "distill0.cooler_B103.port.T"
                    ],
                    "SEQUENCE_LENGTH": 60,
                    "STEP_SIZE": 10,
                    "KFOLDS": 2
                },
                "OPT": {
                    "OPT": "SGD",
                },
                # Insert correct path to experiment folder
                "DIR_EXP": "experiment1_ds6",
                "ID": 1,
                "SEED": 42
            }
        ]

    for hparams in experiments:
        # Prepare TensorBoard logging
        log_dir = os.path.join(hparams['DIR_EXP'], 'tb_log')
        writer = SummaryWriter(log_dir)

        # Ensure experiment directories exist
        create_dir(hparams['DIR_EXP'], hparams['MODEL']['DIR'])
        create_dir(hparams['DIR_EXP'], hparams['TRAINING']['DIR'])

        # Set random seeds for reproducibility
        seed = hparams.get('SEED', 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        logging.info("Experiment hyperparameters: %s", hparams)

        # Select compute device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", device)

        # Initialize data module (supports 'data_siemens' and 'hai' in path)
        data_module = TankDataModule(hparams)

        # Set up training module and run training
        train_module = TrainModule(hparams=hparams, data_module=data_module)
        mean_test, var_test = train_module.training()

        # Prepare hyperparameters and results for TensorBoard
        dir_components = os.path.normpath(hparams['DIR_EXP']).split(os.sep)
        dataset_name = f"{dir_components[-2]}_{dir_components[-1].replace('_', '')}"
        hparam_dict = {
            "DATASET": dataset_name,
            "HIDDEN_DIM": hparams["MODEL"]["HIDDEN_DIM"],
            "N_LAYERS": hparams["MODEL"]["N_LAYERS"],
            "ENCODER_DIM": hparams["MODEL"]["ENCODER_DIM"],
            "DROPOUT": hparams["MODEL"]["DROPOUT"],
            "BATCH_SIZE": hparams["TRAINING"]["BATCH_SIZE"],
            "MAX_EPOCHS": hparams["TRAINING"]["MAX_EPOCHS"],
            "LR": hparams["TRAINING"]["LR"],
            "WEIGHT_DECAY": hparams["TRAINING"]["WEIGHT_DECAY"],
            "NOISE": hparams["TRAINING"]["NOISE"],
            "MASKED": hparams["TRAINING"]["MASKED"],
            "SEQUENCE_LENGTH": hparams["DATA"]["SEQUENCE_LENGTH"],
            "SEED": seed
        }
        metrics = {'mean_test': mean_test, 'var_test': var_test}

        # Log to TensorBoard
        writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metrics)
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment setup and training for sensor clusters"
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        required=True,
        help="Directory containing hparams.json and where outputs are saved."
    )
    args = parser.parse_args()
    main(args.experiment_folder)
