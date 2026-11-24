from dataclasses import dataclass
import torch
import pytorch_lightning as pl
import argparse
import logging
from pytorch_lightning.strategies import DDPStrategy
import os
import json
from typing import Tuple, Optional

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(MODEL_DIR, "..", "params")
os.makedirs(SAVE_DIR, exist_ok=True)


@dataclass
class Config:
    def __init__(self, args, train_shuffle=True):
        """Unified configuration class."""
        # Additional options
        self.loss_mode = args.loss_mode
        self.seed = args.seed
        self.l2_coeff = args.l2_coeff
        self.l3_coeff = args.l3_coeff
        self.l4_coeff = args.l4_coeff

        # Paths
        self.input_dir: str = './data/OSC/'
        self.auxiliary_dir: str = './source/'

        # Spatial configuration
        self.resolution: float = 1 / 12
        self.expand_size: int = 24  # border expansion
        self.rk_mode: str = 'RK4'

        # Time range
        self.dataset_start: str = "1993-01-01"
        self.train_range: str = "93_13"
        self.train_start: str = "1993-01-01"
        self.val_start: str = "2014-01-01"
        self.test_start: str = "2017-01-01"
        self.test_end: str = "2020-01-01"

        # Time configuration
        self.lead: int = 1  # lead time (days)
        self.seq_len: int = 2  # input sequence length (days)
        self.fcst_len: int = 1  # forecast length (days)
        self.output_len = self.fcst_len
        self.land_value: float = 0

        # Training parameters
        self.batch_size: int = 5
        self.learning_rate: float = 1e-4
        self.weight_decay: float = 0.0

        self.train_shuffle = train_shuffle

        # Model parameters
        self.num_var: int = 2  # u and v
        self.in_channels: int = self.num_var * self.seq_len
        self.out_channels: int = self.num_var * self.fcst_len

        # Convergence test
        self.convergence_threshold: float = 3e-6
        self.convergence_patience: int = 3

        # Lightning settings
        self.accumulate_grad_batches: int = 1
        self.gradient_clip_val: float = 1.0
        self.precision: str = "bf16-mixed"

        # Distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            self.strategy = DDPStrategy(find_unused_parameters=True)
        else:
            self.strategy = "auto"

        self.num_nodes: int = 1
        self.sync_batchnorm: bool = True

        self.title = f'GloCur_{self.seq_len}to{self.fcst_len}_{self.train_range}_L{self.loss_mode}'
        self.init_logger()

    @property
    def idx_compensate(self):
        return -self.seq_len - (self.lead - 1) - (self.output_len - 1)

    def init_logger(self):
        logging.basicConfig(format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s',
                            level=logging.INFO,
                            filename=os.path.join(SAVE_DIR, f'{self.title}.log'),
                            filemode='a')
        self.p2log(f'seq_length:{self.seq_len}d\n'
                   f'forecast_length:{self.fcst_len}d\n'
                   f'loss_mode:{self.loss_mode}\n'
                   f'rk_mode:{self.rk_mode}\n'
                   f'title:{self.title}\n'
                   f'Loading dataset configuration...')

    def p2log(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        if os.environ.get('LOCAL_RANK', '0') == '0':
            logging.info(message)
            print(*args, **kwargs)
            logging.getLogger().handlers[0].flush()
