# lightning_modules/lit_flow_module.py
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# Directory containing the current file (lit_module.py)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level, then into params/
SAVE_DIR = os.path.join(MODEL_DIR, "..")
os.makedirs(SAVE_DIR, exist_ok=True)

from model.LossFunc import PhysicsLoss
from utils.metrics import MetricsCalculator
from utils.convergence import ConvergenceDetector
from utils.visualization import plot_loss_curves
from utils.physics import iterate_input


class LitCurrentModule(pl.LightningModule):
    """
    PyTorch Lightning
    """

    def __init__(self, model, data_dict, config):
        super(LitCurrentModule, self).__init__()
        self.config = config
        # model
        self.model = model
        self.register_buffer('land_mask', model.land_mask, persistent=False)
        self.register_buffer('norm_w', model.norm_w, persistent=False)
        self.register_buffer('clim', model.uv_clim, persistent=False)
        self.fcst_len = self.config.fcst_len
        self.seq_len = self.config.seq_len
        self.nv = self.config.num_var
        self.title = self.config.title

        # physic loss
        self.physics_loss = PhysicsLoss(self.config, self.model)

        # metric
        self.metrics_calc = MetricsCalculator(
            clim=self.clim,
            land=self.land_mask,
            norm_w=self.norm_w,
            use_weight=True
        )
        # convergence
        self.convergence_detector = ConvergenceDetector(
            threshold=config.convergence_threshold,
            patience=config.convergence_patience
        )
        # Best validation
        self.best_val_loss = float('inf')
        self.train_loss_history = []
        self.val_loss_history = []
        self.epoch_numbers = []
        self.save_hyperparameters(ignore=['model', 'data_dict', 'land_mask', 'norm_w', 'uv_clim'])

        # optimal
        self.automatic_optimization = True

    def forward(self, t, x):
        """ forward propagate"""
        return self.model(t, x)

    def on_train_start(self):
        # Print parameter counts by module
        if self.trainer.is_global_zero:
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.config.p2log(f"Total trainable parameters: {total_params:,}")
            self.config.p2log("\nParameter breakdown by module:")
            for name, module in self.model.named_children():
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if params > 0:
                    self.config.p2log(f"{name}: {params:,} parameters")

    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            self.config.p2log(f"Starting Epoch {self.current_epoch + 1}")
        # Removed all .zero_() calls since manual buffer accumulation is no longer needed

    def training_step(self, batch, batch_idx):  # return the loss for backward propagate
        t = batch['time']
        uv_seq = batch['input']  # [batch_size, 1*seq_len, lat, lon]
        uv_target = batch['target']  # [batch_size, 1*fcst_len, n_points]

        # Initialize Loss
        loss_dict = {f'l{j + 1}{i + 1}': torch.zeros(1, device=self.device)
                     for j in range(self.fcst_len) for i in range(4)}

        # Predict 1st
        uv_pred = self(t, uv_seq)
        losses = self.physics_loss(
            uv_seq, uv_pred,
            uv_target[:, -2:, :]  # the earliest target
        )

        loss_dict['l11'] = losses['mse']
        if '2' in self.config.loss_mode:
            loss_dict['l12'] = losses['pv'] * 1e05 * self.config.l2_coeff
        if '3' in self.config.loss_mode:
            loss_dict['l13'] = losses['sigma'] * 1e05 * self.config.l3_coeff
        if '4' in self.config.loss_mode:
            loss_dict['l14'] = losses['pearson'] * 1e-5 * self.config.l4_coeff

        # autoregressive
        if self.fcst_len > 1:
            current_t = t.clone()
            current_seq = uv_seq.clone()
            for i in range(self.fcst_len - 1):
                current_t, current_seq = iterate_input(
                    current_t, current_seq, self.nv, uv_pred.detach(), self.seq_len
                )
                uv_pred = self(current_t, current_seq)

                losses = self.physics_loss(
                    current_seq, uv_pred,
                    uv_target[:, -4 - 2 * i:-2 - 2 * i, :],
                )

                loss_dict[f'l{i + 2}1'] = losses['mse']
                if '2' in self.config.loss_mode:
                    loss_dict[f'l{i + 2}2'] = losses['pv'] * 1e05 * self.config.l2_coeff
                if '3' in self.config.loss_mode:
                    loss_dict[f'l{i + 2}3'] = losses['sigma'] * 1e05 * self.config.l3_coeff
                if '4' in self.config.loss_mode:
                    loss_dict[f'l{i + 2}4'] = losses['pearson'] * 1e-5 * self.config.l4_coeff
                    

        total_loss = (
                             self.model.sigma1 * torch.sign(loss_dict['l11']) +
                             self.model.sigma2 * torch.sign(loss_dict['l12']) +
                             self.model.sigma3 * torch.sign(loss_dict['l13']) +
                             self.model.sigma4 * torch.sign(loss_dict['l14'])
                     ) + (
                             torch.exp(-self.model.sigma1) * loss_dict['l11'] +
                             torch.exp(-self.model.sigma2) * loss_dict['l12'] +
                             torch.exp(-self.model.sigma3) * loss_dict['l13'] +
                             torch.exp(-self.model.sigma4) * loss_dict['l14']
                     )
        for i in range(1, self.fcst_len):
            total_loss = total_loss + (
                    torch.exp(-self.model.sigma1) * loss_dict[f'l{i + 1}1'] +
                    torch.exp(-self.model.sigma2) * loss_dict[f'l{i + 1}2'] +
                    torch.exp(-self.model.sigma3) * loss_dict[f'l{i + 1}3'] +
                    torch.exp(-self.model.sigma4) * loss_dict[f'l{i + 1}4'] 
            )

        # === Logging section (keep sync_dist=True; Lightning handles cross-device averaging automatically) ===
        self.log('1d train loss 1', loss_dict['l11'].item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('1d train loss 2', loss_dict['l12'].item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('1d train loss 3', loss_dict['l13'].item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('1d train loss 4', loss_dict['l14'].item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('sigma1', self.model.sigma1.item(), on_epoch=True, sync_dist=True)
        self.log('sigma2', self.model.sigma2.item(), on_epoch=True, sync_dist=True)
        self.log('sigma3', self.model.sigma3.item(), on_epoch=True, sync_dist=True)
        self.log('sigma4', self.model.sigma4.item(), on_epoch=True, sync_dist=True)

        for k in range(self.fcst_len - 1):
            self.log(f'{k + 2}d train loss 1', loss_dict[f'l{k + 2}1'].item(), on_epoch=True, sync_dist=True)
            
        return total_loss

    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            self.model.get_memory_usage()

        # === Removed all manual all_gather logic; read directly from callback_metrics ===
        if self.trainer.is_global_zero:
            for k in range(self.fcst_len):
                # Get loss 1 for the current step
                metric_name_1 = '1d train loss 1' if k == 0 else f'{k + 1}d train loss 1'
                l1 = self.trainer.callback_metrics.get(metric_name_1)
                l1_val = l1.item() if l1 is not None else 0.0

                # Note: only loss 1 for k>0 was logged in training_step, not losses 2 and 3.
                # Fetch defensively here to preserve the original print format.
                l2 = self.trainer.callback_metrics.get(f'{k + 1}d train loss 2')
                l3 = self.trainer.callback_metrics.get(f'{k + 1}d train loss 3')
                l4 = self.trainer.callback_metrics.get(f'{k + 1}d train loss 4')
                l2_val = l2.item() if l2 is not None else 0.0
                l3_val = l3.item() if l3 is not None else 0.0
                l4_val = l4.item() if l4 is not None else 0.0

                self.config.p2log(f'{k + 1}d avg Train loss 1: {l1_val:.7f}\n')
                self.config.p2log(f'{k + 1}d avg Train loss 2: {l2_val:.16f}\n')
                self.config.p2log(f'{k + 1}d avg Train loss 3: {l3_val:.16f}\n')
                self.config.p2log(f'{k + 1}d avg Train loss 4: {l4_val:.16f}\n')

            s1 = self.trainer.callback_metrics.get('sigma1')
            s2 = self.trainer.callback_metrics.get('sigma2')
            s3 = self.trainer.callback_metrics.get('sigma3')
            s4 = self.trainer.callback_metrics.get('sigma4')
            self.config.p2log(f'Model sigma 1: {s1.item() if s1 is not None else 0}\n')
            self.config.p2log(f'Model sigma 2: {s2.item() if s2 is not None else 0}\n')
            self.config.p2log(f'Model sigma 3: {s3.item() if s3 is not None else 0}\n')
            self.config.p2log(f'Model sigma 4: {s4.item() if s4 is not None else 0}\n')

            if self.current_epoch > 2:
                train_1d = self.trainer.callback_metrics.get('1d train loss 1')
                if train_1d is not None:
                    self.train_loss_history.append(train_1d.item())

    def on_validation_epoch_start(self):
        if self.trainer.is_global_zero:
            self.config.p2log(f"Validation {self.current_epoch + 1}")
        # Removed .zero_() operations

    def validation_step(self, batch, batch_idx):
        t = batch['time']
        uv_seq = batch['input']
        uv_target = batch['target']
        val_loss = torch.zeros(self.fcst_len)

        # 1st prediction
        uv_pred = self(t, uv_seq)  # B * 1 * H * W
        lat_w_mse = (self.norm_w * (uv_pred[..., ~self.land_mask] - uv_target[:, -2:, :]) ** 2).sum(dim=-1)
        val_loss[0] = lat_w_mse.mean()

        # iterate prediction
        if self.fcst_len > 1:
            current_t = t.clone()
            current_seq = uv_seq.clone()
            for i in range(self.fcst_len - 1):
                current_t, current_seq = iterate_input(current_t, current_seq, self.nv, uv_pred.detach(), self.seq_len)
                uv_pred = self(current_t, current_seq)
                lat_w_mse = (self.norm_w * (uv_pred[..., ~self.land_mask] -
                                            uv_target[:, -4 - 2 * i:-2 - 2 * i, :]) ** 2).sum(dim=-1)
                val_loss[i + 1] = lat_w_mse.mean()

        # === Keep logging, remove manual accumulation ===
        for k in range(self.fcst_len):
            self.log(f'{k + 1}d val loss', val_loss[k].item(), on_epoch=True, sync_dist=True)

        return val_loss

    def on_validation_epoch_end(self):
        """Operations at the end of the validation epoch."""
        # === Removed manual all_gather logic; read directly from callback_metrics ===
        val_1d = self.trainer.callback_metrics.get('1d val loss')

        if self.trainer.is_global_zero:
            self.config.p2log(f'1d Best Val loss: {self.best_val_loss:.7f}\n')
            for k in range(self.fcst_len):
                avg_val = self.trainer.callback_metrics.get(f'{k + 1}d val loss')
                if avg_val is not None:
                    self.config.p2log(f'{1 + k}d avg Val loss: {avg_val.item():.7f}\n')

        # Determine the best model
        if val_1d is not None and val_1d.item() < self.best_val_loss:
            self.best_val_loss = val_1d.item()
            torch.save(self.model.state_dict(), os.path.join(SAVE_DIR, f"params/{self.title}.pth"))
            # if self.current_epoch < 20:
            #     torch.save(self.model.state_dict(), os.path.join(SAVE_DIR, f"20epoch/{self.title}_20epoch.pth"))
            # if self.current_epoch < 40:
            #    torch.save(self.model.state_dict(), os.path.join(SAVE_DIR, f"40epoch/{self.title}_40epoch.pth"))

        if self.trainer.is_global_zero:
            if self.current_epoch > 2 and val_1d is not None:
                self.val_loss_history.append(val_1d.item())
                self.epoch_numbers.append(self.current_epoch + 1)

    def on_test_start(self):
        pass  # Cleared the original .zero_() logic

    def test_step(self, batch, batch_idx):
        """Test step."""
        t = batch['time']
        uv_seq = batch['input']
        uv_target = batch['target']
        test_loss = torch.zeros(self.fcst_len)

        # 1st predict
        uv_pred = self(t, uv_seq)  # B * 1 * H * W
        lat_w_mse = (self.norm_w * (uv_pred[..., ~self.land_mask] - uv_target[:, -2:, :]) ** 2).sum(dim=-1)
        test_loss[0] = lat_w_mse.mean()

        # later predict
        current_t = t.clone()
        current_seq = uv_seq.clone()
        for i in range(self.fcst_len - 1):
            current_t, current_seq = iterate_input(
                current_t, current_seq, self.nv, uv_pred.detach(), self.seq_len
            )
            uv_pred = self(current_t, current_seq)
            lat_w_mse = (self.norm_w * (
                    uv_pred[..., ~self.land_mask] - uv_target[:, -4 - 2 * i: -2 - 2 * i, :]) ** 2).sum(dim=-1)
            test_loss[i + 1] = lat_w_mse.mean()

        # === Keep logging, remove manual accumulation ===
        for k in range(self.fcst_len):
            self.log(f'{k + 1}d test loss', test_loss[k].item(), on_epoch=True, sync_dist=True)

        return test_loss

    def on_test_end(self):
        # === Removed manual all_gather logic; read directly from callback_metrics ===
        if self.trainer.is_global_zero:
            for k in range(self.fcst_len):
                avg_test = self.trainer.callback_metrics.get(f'{k + 1}d test loss')
                if avg_test is not None:
                    self.config.p2log(f'Avg Test loss {k + 1}d: {avg_test.item():.7f}\n')
          
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate sigma parameters from other parameters
        sigma_params = [p for n, p in self.model.named_parameters()
                        if 'sigma' in n]
        if self.trainer.is_global_zero:
            print(sigma_params)
        other_params = [p for n, p in self.model.named_parameters()
                        if 'sigma' not in n]
        optimizer = optim.AdamW([
            {'params': other_params, 'lr': self.config.learning_rate},
            {'params': sigma_params, 'lr': self.config.learning_rate}
        ], weight_decay=self.config.weight_decay)

        return [optimizer]

    def on_train_end(self):
        """Operations at the end of training."""
        if self.trainer.is_global_zero:
            if len(self.train_loss_history) > 0:
                plot_loss_curves(
                    self.train_loss_history,
                    self.val_loss_history,
                    self.epoch_numbers,
                    self.config.title
                )
                # 2. Log training completion
            self.config.p2log("Training stage finished.\n")