import os
from pathlib import Path
import gc
import logging
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from dataModule.dataModule import GenericDataModule  # Assuming GenericDataModule exists

logger = logging.getLogger(__name__)


class GenericDataModule:
    """
    Base class for loading and preprocessing tabular time-series data.
    Supports various file formats and simple noise injection.
    """
    def __init__(self, hparams: dict):
        self.hparams = hparams
        cfg = hparams.get('DATA', {})
        required = ['DIR']
        self.check_hparams(required_keys=required_keys)

        self.path = Path(cfg['DIR'])
        self.columns = cfg.get('COLUMNS', None)

    def check_hparams(self, required_keys):
        # Check for missing hyperparameters
        missing_keys = [key for key in required_keys if key not in self.hparams.get('DATA', {})]
        if missing_keys:
            # Log an error with the missing keys
            logger.error(f"check_hparams: Missing hyperparameters: {missing_keys}")
            
            # Raise an exception to halt execution
            raise ValueError(f"Missing hyperparameters: {missing_keys}")

    def load_data(self) -> pd.DataFrame:
        """
        Load raw data file into a DataFrame based on extension.
        Supports CSV, Parquet, Excel, JSON, TSV, Feather, ORC, and Pickle.

        Returns:
            pd.DataFrame: Loaded dataset with optional column filtering.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")

        ext = self.path.suffix.lower()
        read_kwargs = {'usecols': self.columns} if self.columns else {}

        try:
            if ext == '.csv':
                df = pd.read_csv(self.path, **read_kwargs)
            elif ext in ['.parquet', '.pq']:
                df = pd.read_parquet(self.path, columns=self.columns)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(self.path, **read_kwargs)
            elif ext == '.json':
                df = pd.read_json(self.path)
                if self.columns:
                    df = df[self.columns]
            elif ext in ['.tsv', '.txt']:
                df = pd.read_csv(self.path, sep='\t', **read_kwargs)
            elif ext == '.feather':
                df = pd.read_feather(self.path)
                if self.columns:
                    df = df[self.columns]
            elif ext == '.orc':
                df = pd.read_orc(self.path)
                if self.columns:
                    df = df[self.columns]
            elif ext == '.pkl':
                df = pd.read_pickle(self.path)
                if self.columns:
                    df = df[self.columns]
            else:
                raise ValueError(f"Unsupported extension '{ext}'")
        except Exception as e:
            logger.exception('Failed to load data: %s', e)
            raise

        return df

    def add_noise(
        self,
        df: pd.DataFrame,
        noise_fraction: float = 0.1,
        clip: bool = True,
        multiply: bool = False
    ) -> pd.DataFrame:
        """
        Inject Gaussian noise into numeric columns.

        Args:
            df: Input DataFrame (modified in place).
            noise_fraction: Max noise fraction relative to data range.
            clip: Clamp noise to [-noise_fraction, +noise_fraction].
            multiply: Scale noise by original values if True.

        Returns:
            pd.DataFrame: Noisy DataFrame.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns
        sigma = noise_fraction / 3
        noise = np.random.normal(0, sigma, size=df[num_cols].shape)
        if clip:
            noise = np.clip(noise, -noise_fraction, noise_fraction)
        if multiply:
            df[num_cols] += df[num_cols].values * noise
        else:
            df[num_cols] += noise
        return df

    def scale(
        self,
        df: pd.DataFrame,
        scaler_dir: str
    ) -> np.ndarray:
        """
        Fit a MinMax scaler on the DataFrame and save it.

        Returns scaled NumPy array.
        """
        scaler = MinMaxScaler()
        arr = scaler.fit_transform(df)
        joblib.dump(scaler, os.path.join(scaler_dir, 'scaler.gz'))
        logger.debug('Scaler saved to %s', scaler_dir)
        return arr

    def to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert NumPy array to float32 torch.Tensor."""
        return torch.from_numpy(arr.astype(np.float32))

    def init_dataloader(
        self,
        data: list,
        batch_size: int,
        shuffle: bool = False
    ) -> DataLoader:
        """Wrap list of tensors into a DataLoader."""
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=True)


class TankDataModule(GenericDataModule):
    """
    DataModule for fixed-length time-series sequences and k-fold splits.
    """
    def __init__(self, hparams: dict):
        super().__init__(hparams)
        cfg = hparams['DATA']
        required = ['SEQUENCE_LENGTH', 'STEP_SIZE', 'KFOLDS']
        self.check_hparams(required)

        self.seq_len = cfg['SEQUENCE_LENGTH']
        self.step = cfg['STEP_SIZE']
        self.kfold = KFold(n_splits=cfg['KFOLDS'], shuffle=True, random_state=hparams['SEED'])

        # Load, scale, and split data
        df = self.load_data()
        arr = self.scale(df, hparams['DIR_EXP'])
        samples = self._make_sequences(arr)
        train_count = int(0.9 * len(samples))
        self.train_samples = samples[:train_count]
        self.test_samples = samples[train_count:]

    def _make_sequences(self, arr: np.ndarray) -> list:
        """Generate overlapping sequences from full array."""
        seqs = []
        for i in range(0, len(arr) - self.seq_len + 1, self.step):
            seqs.append(self.to_tensor(arr[i:i+self.seq_len]))
        return seqs

    def get_fold_dataloaders(self, fold_index: int):
        """
        Return train and validation DataLoaders for a k-fold split.
        """
        # Prepare array of samples
        samples = torch.stack(self.train_samples)
        splits = list(self.kfold.split(samples))
        train_idx, val_idx = splits[fold_index-1]
        train_data = [samples[i] for i in train_idx]
        val_data = [samples[i] for i in val_idx]

        batch = self.hparams['TRAINING']['BATCH_SIZE']
        return (
            self.init_dataloader(train_data, batch, shuffle=True),
            self.init_dataloader(val_data, batch)
        )

    def get_test_dataloader(self) -> DataLoader:
        """Return DataLoader for the held-out test samples."""
        batch = self.hparams['TRAINING']['BATCH_SIZE']
        return self.init_dataloader(self.test_samples, batch)
