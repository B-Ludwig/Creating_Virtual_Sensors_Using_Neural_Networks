import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt


double = float

class GradientSearch:
    """
    Performs per-sample gradient-based optimization on input tensors to
    minimize reconstruction loss under a fixed model.
    """
    def __init__(self, hparam: dict, model: nn.Module):
        self.hparam = hparam
        self.model = model
        # Use per-element MSE without reduction for sample-wise loss
        self.loss_fn = nn.MSELoss(reduction='none')
        # Default optimizer settings if not provided
        self.hparam.setdefault("OPT", {}).setdefault("OPT", "SGD")
        self.hparam.setdefault("OPT", {}).setdefault('LR', 1e-2)
        self.hparam.setdefault("OPT", {}).setdefault('MOMENTUM', 0.9)

    def step_batch(
        self,
        x_guess: torch.Tensor,
        y_guess: torch.Tensor,
        opt_vars: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Execute one optimization step on a batch of inputs.

        Args:
            x_guess: Initial input tensor [B, T, D], requires_grad.
            y_guess: Target output tensor [B, T, D].
            opt_vars: Boolean mask [B, T, D]; True entries are optimized.

        Returns:
            loss_y: Per-sample output losses [B].
            x_opt: Updated input tensor detached [B, T, D].
            y_hat: Model reconstruction [B, T, D].
        """
        device = x_guess.device
        B = x_guess.size(0)

        # Choose optimizer based on hyperparameter
        if self.hparam['OPT']['OPT'] == 'SGD':
            optimizer = optim.SGD([x_guess], lr=self.hparam['OPT']['LR'])
        elif self.hparam['OPT']['OPT'] == 'SGD-nesterov':
            optimizer = optim.SGD(
                [x_guess],
                lr=self.hparam['OPT']['LR'],
                momentum=self.hparam['OPT']['MOMENTUM'],
                nesterov=True
            )
        elif self.hparam['OPT']['OPT'] == 'ADAM':
            optimizer = optim.Adam([x_guess], lr=self.hparam['OPT']['LR'])
        elif self.hparam['OPT']['OPT'] == 'ASGD':
            optimizer = optim.ASGD([x_guess], lr=self.hparam['OPT']['LR'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparam['OPT']['OPT']}")

        optimizer.zero_grad()

        # Forward pass through model
        self.model.train()
        self.model.init_weights(B, gpu=(device.type=='cuda'))
        y_hat = self.model(x_guess)

        # Compute masked MSE loss per sample
        mask = (~opt_vars).float()
        loss_elem = self.loss_fn(y_hat * mask, y_guess * mask)
        # Sum over time and feature dims, normalize
        loss_y = (loss_elem * mask).sum(dim=(1,2)) / (mask.sum(dim=(1,2)) + 1e-8)

        # Backpropagate summed loss
        total_loss = loss_y.sum()
        total_loss.backward()

        # Normalize gradients per sample to unit norm
        grads = x_guess.grad.view(B, -1)
        norm = grads.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x_guess.grad = (grads / norm).view_as(x_guess)

        # Update guess tensor in-place
        optimizer.step()
        return loss_y, x_guess.detach(), y_hat.detach()


class OptModuleBatch:
    """
    High-level optimizer managing iterative reconstruction for missing inputs.
    """
    def __init__(self, hparams: dict, model: nn.Module):
        self.hparam = hparams
        self.model = model
        self.GS = GradientSearch(hparams, model)

    def find_params(
        self,
        x_guess: torch.Tensor,
        y_guess: torch.Tensor,
        opt_vars: torch.Tensor,
        max_cycles: int = 5000,
        patience: int = 100,
        max_consecutive_cycles: int = 200,
        init_window: int = 100
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, int):
        """
        Iteratively optimize masked inputs until convergence criteria per sample.

        Args:
            x_guess: Initial input [B, T, D] with requires_grad.
            y_guess: Ground truth [B, T, D].
            opt_vars: Bool mask [B, T, D] indicating variables to optimize.
            max_cycles: Global iteration cap.
            patience: No-improvement stops after this many cycles.
            max_consecutive_cycles: Delta stagnation cap.
            init_window: Warm-up iterations to compute thresholds.

        Returns:
            x_hat: Final optimized inputs.
            y_hat: Final reconstructions.
            y_loss_b: Per-sample final losses.
            n_cycles: Number of iterations executed.
        """
        device = x_guess.device
        B = x_guess.size(0)

        # State tracking per sample
        best_loss = torch.full((B,), float('inf'), device=device)
        no_improve = torch.zeros(B, dtype=torch.int, device=device)
        consec = torch.zeros(B, dtype=torch.int, device=device)
        converged = torch.zeros(B, dtype=torch.bool, device=device)
        loss_old = torch.full((B,), float('inf'), device=device)

        # Collect initial losses for thresholding
        init_losses = []
        min_delta = 0.0
        threshold = 0.0

        # Prepare x_guess for gradients
        x_guess = x_guess.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([x_guess], lr=self.hparam.get('LR', 1e-2))

        pbar = tqdm(total=max_cycles, desc='Optimizing batch')
        n_cycles = 0
        start_time = time.perf_counter()

        # Iterative optimization loop
        while n_cycles < max_cycles and not converged.all():
            optimizer.zero_grad()

            # Single optimization step
            y_loss_b, x_hat, y_hat = self.GS.step_batch(
                x_guess, y_guess, opt_vars
            )

            # During init window, store average loss
            if n_cycles < init_window:
                init_losses.append(y_loss_b.mean().item())
            if n_cycles == init_window:
                mean_init = np.mean(init_losses)
                min_delta = 0.01 * mean_init
                threshold = 0.01 * min_delta

            # Compute improvement metrics
            delta = (loss_old - y_loss_b).abs()
            loss_old = y_loss_b.clone()

            if n_cycles >= init_window:
                # Update best-loss and no-improvement counter
                improved = y_loss_b < (best_loss - min_delta)
                best_loss[improved] = y_loss_b[improved]
                no_improve[~improved] += 1

                # Consecutive cycles without significant delta
                stalled = delta <= threshold
                consec[stalled] += 1
                consec[~stalled] = 0

                # Mark samples as converged
                converged |= (no_improve >= patience) | (consec >= max_consecutive_cycles)

            # Zero grads for converged samples to freeze them
            with torch.no_grad():
                x_guess.grad[converged] = 0

            optimizer.step()
            n_cycles += 1
            pbar.update(1)

        pbar.close()
        duration = time.perf_counter() - start_time
        print(f"Finished {n_cycles} iterations in {duration:.1f}s; "
              f"{converged.sum().item()}/{B} converged.")

        return x_hat, y_hat, y_loss_b, n_cycles
