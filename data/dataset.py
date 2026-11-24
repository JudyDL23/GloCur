# data/dataset.py
import pytorch_lightning as pl
import glob
import os
from datetime import datetime, timedelta
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random


def seed_worker(worker_id):
    """Worker initialization function for DataLoader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def time_coding(nowtime):
    """Time encoding function."""
    base = datetime(1959, 1, 1)
    now_minus_base = nowtime - base
    time_now_base = now_minus_base.days
    now_time_code = 12 + time_now_base * 24
    return now_time_code


def normalize(data, mask, landvalue):
    """Normalize ocean region data."""
    data_flatten = torch.flatten(data, start_dim=1)  # keep bs
    max_val, _ = torch.max(data_flatten, dim=1)  # keep bs
    min_val, _ = torch.min(data_flatten, dim=1)  # keep bs
    norm_data = torch.zeros_like(data)
    for i in range(len(data)):
        norm_data[i, ...] = (data[i, ...] - min_val[i]) / (max_val[i] - min_val[i])
    norm_data[..., mask] = landvalue
    return norm_data


class FlowDataset(Dataset):
    """Flow field dataset."""

    def __init__(self, file_list, land_mask, config):
        self.file_list = file_list
        self.seq_len = config.seq_len
        self.fcst_len = config.fcst_len
        self.output_len = config.output_len
        self.mask = land_mask
        self.landv = config.land_value
        self.total_days = len(file_list)
        self.max_start_idx = max(self.total_days + config.idx_compensate, 0)
        self.lead = config.lead

    def __len__(self):
        return self.max_start_idx

    def __getitem__(self, idx):
        if idx >= self.max_start_idx:
            raise IndexError(f'Index {idx} out of range! Max index: {self.max_start_idx - 1}')

        # Input sequence
        input_data = []
        for i in range(self.seq_len):
            file_path = self.file_list[idx + i]
            data = self._load_file(file_path)
            input_data.append(data)

        # Extract date information for time encoding
        date_str = self.file_list[idx + self.seq_len - 1].split('/')[-1].split('.')[0].split('_')[1]
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        t = time_coding(date_obj)  #

         # Label
        target_idx = idx + self.seq_len + self.lead - 1
        if target_idx + self.output_len - 1 >= len(self.file_list):
            raise IndexError("Target index out of range!")

        target_data = []
        for i in range(self.output_len):
            file_path = self.file_list[target_idx + i]
            data = self._load_file(file_path)
            target_data.append(data[..., ~self.mask])
            
        # turn into tensor
        input_tensor = torch.stack(input_data[::-1]).flatten(0, 1)
        # (2 * seq_len, H, W)  (u2, v2, u1, v1, u0, v0 ...)
        target_tensor = torch.stack(target_data[::-1]).flatten(0, 1)
        # (2 * fcst_len, N_points)  (u2, v2, u1, v1, u0, v0 ...)

        return {
            'time': torch.tensor(t, dtype=torch.float),
            'input': input_tensor,
            'target': target_tensor,
            'idx': idx  # optional, for debugging
        }

    def _load_file(self, file_path):
        """Load a single npy file."""
        uv_slice = np.load(file_path)[..., 1:, :]  # u,v
        uv_slice[..., self.mask] = self.landv
        mask = np.isnan(uv_slice) | (uv_slice < -10)  # 2, H, W  remove the outliers -32770
        uv_slice[..., mask] = self.landv
        return torch.from_numpy(uv_slice).float()  # 2041, 4320 ----> 2040 * 4320


class GlobalCurrentDataModule(pl.LightningDataModule):
    def __init__(self, config, land_mask):
        super().__init__()
        self.config = config
        self.land_mask = land_mask
        self.seed = config.seed
        self.bs = config.batch_size
        self.pin_memory = True
        self.train_start = datetime.strptime(config.train_start, "%Y-%m-%d")
        self.val_start = datetime.strptime(config.val_start, "%Y-%m-%d")
        self.test_start = datetime.strptime(config.test_start, "%Y-%m-%d")
        self.test_end = datetime.strptime(config.test_end, "%Y-%m-%d")
        self.base = datetime.strptime(config.dataset_start, "%Y-%m-%d")

        self.train_start_id = (self.train_start - self.base).days
        self.val_start_id = (self.val_start - self.base).days
        self.test_start_id = (self.test_start - self.base).days
        self.test_end_id = (self.test_end - self.base).days

        self.idx_cps = config.idx_compensate

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        if torch.cuda.device_count() > 1:
          self.drp = True
        else:
          self.drp = False

    def prepare_data(self):  # avoid the same process running in all the devices
        in_path = self.config.input_dir
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Input File path does not Exist! Path: {in_path}")

    def setup(self, stage=None):
        in_path = self.config.input_dir
        # obtain all the NPY files
        npy_files = sorted(glob.glob(os.path.join(in_path, "uv0_*.npy")))  # "XXXX_20020123.npy"
        if not npy_files:
            raise FileNotFoundError(f"No NPY files found in {in_path}")

        # print(f"Found {len(npy_files)} NPY files")
        # print(f"First few files: {npy_files[:1]}")
        # print(f"Last few files: {npy_files[-1:]}")

        # Dataset Division
        train_files = npy_files[max(self.train_start_id + self.idx_cps, 0):self.val_start_id]
        val_files = npy_files[self.val_start_id + self.idx_cps:self.test_start_id]
        test_files = npy_files[self.test_start_id + self.idx_cps:self.test_end_id]

        # print(f"Training samples: {len(train_files)}")
        # print(f"Validation samples: {len(val_files)}")
        # print(f"Test samples: {len(test_files)}")

        # Create dataset
        self.train_dataset = FlowDataset(train_files, self.land_mask, self.config)
        self.val_dataset = FlowDataset(val_files, self.land_mask, self.config)
        self.test_dataset = FlowDataset(test_files, self.land_mask, self.config)

    def train_dataloader(self):
        # Create Dataloader
        if self.config.train_shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            return DataLoader(
                self.train_dataset,
                batch_size=self.bs,
                shuffle=True,
                num_workers=4,
                pin_memory=self.pin_memory,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.bs,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.bs,
            shuffle=False,
            num_workers=2,
            pin_memory=self.pin_memory,
            drop_last=self.drp
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.bs,
            shuffle=False,
            num_workers=2,
            pin_memory=self.pin_memory,
            drop_last=self.drp
          
        )
