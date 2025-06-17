import os
import gc
import math
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataModule.dataModule import TankDataModule
from model.net import LSTMModel
from utils.utils import create_dir

logger = logging.getLogger(__name__)


class TrainModule:
    """
    Encapsulates training, validation, and testing loops for sequence models.
    """
    def __init__(self, hparams: dict, data_module: TankDataModule):
        """
        Initialize training settings and validate hyperparameters.

        Args:
            hparams: Configuration dict with TRAINING and MODEL settings.
            data_module: DataModule providing data loaders.
        """
        self.hparams = hparams
        self.data_module = data_module

        required = ['DIR', 'BATCH_SIZE', 'LR', 'WEIGHT_DECAY', 'MAX_EPOCHS', 'MASKED']
        self.check_hparams(required)

        tcfg = hparams['TRAINING']
        self.dir = tcfg['DIR']
        self.batch_size = tcfg['BATCH_SIZE']
        self.lr = tcfg['LR']
        self.weight_decay = tcfg['WEIGHT_DECAY']
        self.max_epochs = tcfg['MAX_EPOCHS']
        self.masked = tcfg['MASKED'] == 'TRUE'

    def mask_input(self, x: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Randomly mask up to 5 sensor channels after a warm-up period.

        Args:
            x: Input tensor (B, T, D).
            epoch: Current epoch index.

        Returns:
            Potentially masked copy of x.
        """
        if epoch < 250:
            return x
        num_sensors = x.size(-1)
        k = random.randint(1, min(5, math.ceil(num_sensors / 5)))
        channels = random.sample(range(num_sensors), k)
        x_masked = x.clone()
        x_masked[:, :, channels] = -1.0
        return x_masked

    def compute_loss(self, model: nn.Module, x_in: torch.Tensor, device: str):
        """
        Compute mean squared reconstruction loss.

        Args:
            model: Sequence model with a forward method.
            x_in: Input tensor to reconstruct.
            device: Device string ('cpu' or 'cuda').

        Returns:
            Tuple of (loss_value, model_output).
        """
        x = x_in.to(device)
        model.init_weights(batch_size=x.size(0), gpu=(device=='cuda'))
        output = model(x)
        loss = nn.functional.mse_loss(output, x)
        return loss, output

    def train_step(self, loader, model: nn.Module, optimizer, device: str, epoch: int) -> float:
        """
        Perform one training epoch.

        Args:
            loader: Training DataLoader.
            model: Model to train.
            optimizer: Optimizer instance.
            device: Computation device.
            epoch: Current epoch index.

        Returns:
            Average training loss.
        """
        model.train()
        total = 0.0
        for batch in loader:
            x_raw = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_in = self.mask_input(x_raw, epoch) if self.masked else x_raw

            optimizer.zero_grad()
            loss, _ = self.compute_loss(model, x_in, device)
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    def val_step(self, loader, model: nn.Module, device: str, epoch: int) -> float:
        """
        Compute validation loss for one epoch.
        """
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                x_raw = batch[0] if isinstance(batch, (list, tuple)) else batch
                x_in = self.mask_input(x_raw, epoch) if self.masked else x_raw

                loss, _ = self.compute_loss(model, x_in, device)
                total += loss.item()
        return total / len(loader)

    def test_step(self, loader, model: nn.Module, device: str) -> float:
        """
        Evaluate test loss without masking.
        """
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                loss, _ = self.compute_loss(model, x, device)
                total += loss.item()
        return total / len(loader)

    def training(self) -> (float, float):
        """
        Execute k-fold training, logging to TensorBoard and saving best models.

        Returns:
            Mean and variance of test losses across folds.
        """
        # Seed and device setup
        seed = self.hparams['SEED']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        test_losses = []
        folds = self.hparams['DATA']['KFOLDS']

        for fold in range(folds):
            # Model instantiation
            if self.hparams['MODEL']['NAME'] != 'LSTMModel':
                logger.warning('Unsupported model: %s', self.hparams['MODEL']['NAME'])
                return
            model = LSTMModel(hparams=self.hparams).to(device)

            # Prepare TensorBoard writer on first fold
            if fold == 0:
                folder = model.get_full_model_name()
                path = os.path.join(self.hparams['DIR_EXP'], 'tb_log', folder)
                create_dir(self.hparams['DIR_EXP'], f'tb_log/{folder}')
                self.writer = SummaryWriter(path)

            # Data loaders and optimizer
            dl_train, dl_val = self.data_module.get_fold_dataloaders(fold+1)
            dl_test = self.data_module.get_test_dataloader()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            # Epoch loop
            for epoch in tqdm(range(self.max_epochs), desc=f'Fold {fold}'):  
                train_loss = self.train_step(dl_train, model, optimizer, device, epoch)
                self.writer.add_scalar(f'train/fold_{fold}', train_loss, epoch)

                if epoch % 5 == 0 and epoch > 0:
                    val_loss = self.val_step(dl_val, model, device, epoch)
                    self.writer.add_scalar(f'val/fold_{fold}', val_loss, epoch)
                    tqdm.write(f'Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}')

            # Final test evaluation
            test_loss = self.test_step(dl_test, model, device)
            self.writer.add_scalar(f'test/fold_{fold}', test_loss, self.max_epochs)
            test_losses.append(test_loss)

            logger.info('Fold %d losses: train=%.4f val=%.4f test=%.4f', fold, train_loss, val_loss, test_loss)
            model.save_model(fold, seed)

            # Cleanup
            self.writer.close()
            gc.collect()
            torch.cuda.empty_cache()

        mean_test = float(np.mean(test_losses))
        var_test = float(np.var(test_losses))
        logger.info('Mean test loss=%.4f var=%.4f', mean_test, var_test)
        return mean_test, var_test

    def testing(self) -> (float, float):
        """
        Load a saved model and compute test loss on full set.

        Returns:
            Mean and variance of test loss.
        """
        seed = self.hparams['SEED']
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        dl_test = self.data_module.get_test_dataloader()
        model = LSTMModel.load_model(self.hparams, fold=0).to(device)

        test_loss = self.test_step(dl_test, model, device)
        return float(test_loss), 0.0

    def check_hparams(self, required_keys):
        """
        Ensure required TRAINING hyperparameters exist.
        """
        missing = [k for k in required_keys if k not in self.hparams.get('TRAINING', {})]
        if missing:
            logger.error('Missing training hyperparameters: %s', missing)
            raise ValueError(f'Missing hyperparameters: {missing}')
