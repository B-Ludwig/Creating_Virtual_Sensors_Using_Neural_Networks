import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict
import logging
from utils.utils import create_dir

logger = logging.getLogger(__name__)


class GenericModel(nn.Module, ABC):
    """
    Abstract base class for ML models, handling parameter checks and saving.
    """
    def __init__(self, hparams: Dict):
        super().__init__()
        self.hparams = hparams

        # Ensure essential MODEL keys are present
        required = ['DIR', 'NAME']
        self.check_hparams(required)

        self.dir = hparams['MODEL']['DIR']
        self.name = hparams['MODEL']['NAME']

    def save_model(self, fold: int, seed: int):
        """
        Persist model weights to disk, organizing by fold and seed.

        Args:
            fold: Cross-validation fold identifier.
            seed: Random seed identifier.
        """
        base_path = os.path.join(self.hparams['DIR_EXP'], self.dir)
        create_dir(base_path, self.get_full_model_name())
        filename = f"{self.get_full_model_name()}_Fold_{fold}_Seed_{seed}.pth"
        path = os.path.join(base_path, self.get_full_model_name(), filename)
        torch.save(self.state_dict(), path)
        logger.info('Model saved to %s', path)

    def check_hparams(self, required_keys):
        """
        Validate presence of required MODEL hyperparameters.

        Args:
            required_keys: List of keys expected under hparams['MODEL'].
        """
        missing = [k for k in required_keys if k not in self.hparams.get('MODEL', {})]
        if missing:
            logger.error('Missing hyperparameters: %s', missing)
            raise ValueError(f"Missing hyperparameters: {missing}")

    @classmethod
    @abstractmethod
    def load_model(cls, hparams: Dict, *args, **kwargs) -> 'GenericModel':
        """
        Instantiate and load model weights from storage.

        Must be implemented by subclasses.
        """
        pass


class LSTMModel(GenericModel):
    """
    Sequence-to-sequence LSTM autoencoder for time-series reconstruction.
    """
    def __init__(self, hparams: Dict):
        super().__init__(hparams)

        # Required RNN hyperparameters
        required = ['INPUT_DIM', 'HIDDEN_DIM', 'N_LAYERS', 'DROPOUT', 'ENCODER_DIM', 'OUTPUT_DIM']
        self.check_hparams(required)

        m = hparams['MODEL']
        self.input_dim = m['INPUT_DIM']
        self.hidden_dim = m['HIDDEN_DIM']
        self.encoder_dim = m['ENCODER_DIM']
        self.output_dim = m['OUTPUT_DIM']
        self.n_layers = m['N_LAYERS']
        self.dropout = m['DROPOUT']

        # Encoder
        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            dropout=self.dropout,
            batch_first=True
        )
        self.fc1 = nn.Linear(self.hidden_dim, self.encoder_dim)
        self.layer_dropout = nn.Dropout(self.dropout)

        # Decoder
        self.fc2 = nn.Linear(self.encoder_dim, self.hidden_dim)
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.encoder_dim,
            num_layers=self.n_layers,
            dropout=self.dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(self.encoder_dim, self.input_dim)

        # Hidden states for both LSTM layers
        self.hidden1 = None
        self.hidden2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM autoencoder.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Reconstructed output of same shape as input.
        """
        if self.hidden1 is None or self.hidden2 is None:
            logger.error('Call init_weights before forward pass')
            return None

        # Encode
        out1, self.hidden1 = self.lstm1(x, self.hidden1)
        # Flatten time and batch dims for linear layer
        batch_size, seq_len, _ = out1.size()
        flat = out1.contiguous().view(-1, self.hidden_dim)
        enc = F.relu(self.fc1(self.layer_dropout(flat)))

        # Decode
        dec = F.relu(self.fc2(self.layer_dropout(enc)))
        dec = dec.view(batch_size, seq_len, self.hidden_dim)
        out2, self.hidden2 = self.lstm2(dec, self.hidden2)
        recon = self.fc_out(out2)
        return recon

    def init_weights(self, batch_size: int, gpu: bool = True):
        """
        Initialize hidden states to zeros for encoder and decoder.

        Args:
            batch_size: Number of sequences per batch.
            gpu: Place tensors on CUDA if True.
        """
        weight = next(self.parameters()).data
        device = 'cuda' if gpu else 'cpu'

        zeros1 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        zeros2 = torch.zeros(self.n_layers, batch_size, self.encoder_dim, device=device)
        self.hidden1 = (zeros1, zeros1)
        self.hidden2 = (zeros2, zeros2)

    @classmethod
    def load_model(cls, hparams: Dict, fold: int) -> 'LSTMModel':
        """
        Load model weights for a specific fold from disk.

        Args:
            hparams: Hyperparameter dictionary.
            fold: Fold identifier used in filename.

        Returns:
            An LSTMModel instance in eval mode.
        """
        model = cls(hparams)
        filename = f"{model.name}_Fold_{fold}.pth"
        path = os.path.join(model.hparams['DIR_EXP'], model.dir, filename)

        try:
            state = torch.load(path, map_location='cpu')
            model.load_state_dict(state)
            model.eval()
            logger.info('Loaded model from %s', path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

        return model

    def get_full_model_name(self) -> str:
        """
        Construct a descriptive model name including key hyperparameters.

        Returns:
            Formatted model name string.
        """
        parts = [
            self.name,
            f"IN{self.input_dim}",
            f"HD{self.hidden_dim}",
            f"ED{self.encoder_dim}",
            f"NL{self.n_layers}",
            f"DO{self.dropout}"
        ]
        return '_'.join(parts)
