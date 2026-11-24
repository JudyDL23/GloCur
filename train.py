import os
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"

import torch
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')

import numpy as np
import random
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar

from model.config import Config
from data.preprocessor import load_and_process_data
from data.dataset import GlobalCurrentDataModule
from model.SwTEarth import SwinTransformer
from lightning_modules.lit_module import LitCurrentModule
from utils.visualization import *


class DetailedProgressBar(TQDMProgressBar):
    def format_metrics(self, metrics):
        # Keep only scalar values.
        metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
        return ", ".join(
            f"{k}: {v}" if isinstance(v, str) else f"{k}: {v:.4e}"
            for k, v in metrics.items()
        )


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def print_gpu_memory():
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    cached = torch.cuda.memory_reserved(0) / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB, Total: {total:.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Global Currents Predictor Training Config")
    parser.add_argument('--loss_mode', type=str, default='1', help='Loss mode (e.g., 1, 2, 3)')
    parser.add_argument('--l2_coeff', type=float, default=1, help='Coefficient for the PV loss')
    parser.add_argument('--l3_coeff', type=float, default=1, help='Coefficient for the deformation loss')
    parser.add_argument('--l4_coeff', type=float, default=0, help='Coefficient for the Pearson loss')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    config = Config(args, True)

    set_all_seeds(config.seed)

    # Load auxiliary data: land mask, SLA climatology, mean/std, and related fields.
    data_dict = load_and_process_data(config)
    print("=== Initializing, GPU State: ===")
    print_gpu_memory()

    # DataModule
    dm = GlobalCurrentDataModule(config, data_dict['land_mask'])
    dm.prepare_data()

    # Model architecture
    model = SwinTransformer(data_dict, config, spemb=True, img_size=(2040, 4320), patch_size=12,
                            embed_dim=192, depths=(4,), bottle_neck=(4,), num_heads=(4, 8, 4),
                            window_size=10, mlp_ratio=4., ape=False, use_checkpoint=False)

    # LightningModule
    lit_current = LitCurrentModule(
        model=model,
        data_dict=data_dict,
        config=config,
    )

    # Training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='./ckpt/',
        filename=config.title,
        save_top_k=2,
        monitor='1d val loss',
        mode='min',
    )

    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=config.title)

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator="auto",
        devices=[0, 1],
        strategy=config.strategy,
        num_nodes=config.num_nodes,
        accumulate_grad_batches=config.accumulate_grad_batches,
        precision=config.precision,
        sync_batchnorm=config.sync_batchnorm,
        callbacks=[DetailedProgressBar(), checkpoint_callback],
        logger=logger,
        log_every_n_steps=1,
        detect_anomaly=False,
    )

    trainer.fit(lit_current, dm)
    config.p2log('Training Completed!')
    trainer.test(lit_current, dm)
