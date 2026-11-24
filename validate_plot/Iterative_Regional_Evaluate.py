import pandas as pd
from tqdm import tqdm
from model.config import Config
import argparse
from torch.utils.data import Dataset, DataLoader
import torch
from data.preprocessor import load_and_process_data
from data.dataset import GlobalCurrentDataModule
from model.SwTEarth import SwinTransformer
from utils.visualization import *
from utils.physics import iterate_input
from utils.metrics import MetricsCalculator
import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import random
from pathlib import Path
from lightning_modules.lit_module import LitCurrentModule


# DataProcessor
parser = argparse.ArgumentParser(description="Regionnal Vel Evaluation")
parser.add_argument('--loss_mode', type=str, default='1', help='Loss functions mode (e.g., 1, 2, 3)')
parser.add_argument('--l2_coeff', type=float, default=1.0, help='Coefficient for loss2 terms')
parser.add_argument('--l3_coeff', type=float, default=1.0, help='Coefficient for loss3 terms')
parser.add_argument('--l4_coeff', type=float, default=1.0, help='Coefficient for loss4 terms')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--title', type=str, default="GloCur_2to1_93_13_L1_25epoch_seed42", help='ckpt file name')
# GloCur_2DCU_no_softmax_2to1_93_13_L123_1_5_0_RK4_twoway_0.5_0.5_25epoch_seed42
# GloCur_2to1_93_13_L1_25epoch_seed42.ckpt
args = parser.parse_args()

device = torch.device('cuda:0')

start_day = 1
iterate_day = 10
day_rank = iterate_day - start_day + 1
config = Config(args, train_shuffle=False)
config.output_len = iterate_day
config.dataset_start = "1993-01-01"
config.train_start = "1993-01-01"
config.val_start = "2014-01-01"
config.test_start = "2017-01-01"
config.test_end = "2020-01-01"
config.batch_size = 4
nv = config.num_var
sql = config.seq_len
data_dict = load_and_process_data(config)

eke_clim_pth = './source/Global_eke_clim_93_13.npy'
save_dir = "./validate_plot/"
eke_clim = torch.from_numpy(np.load(eke_clim_pth)).to(device)
if eke_clim.shape[-2] != 2040:
    eke_clim = eke_clim[1:, :]

model = SwinTransformer(data_dict, config, spemb=True, img_size=(2040, 4320), patch_size=12,
                        embed_dim=192, depths=(4,), bottle_neck=(4,), num_heads=(4, 8, 4),
                        window_size=10, mlp_ratio=4., ape=False, use_checkpoint=False)
title = args.title
litmodel = LitCurrentModule.load_from_checkpoint(
    f'./ckpt/{title}.ckpt',
    map_location=device,
    model=model,           # pass any additional arguments if needed
    data_dict=data_dict,   # pass constructor arguments at load time
    config=config,
    strict=False
)
model = litmodel.model.to(device)

# --- four regions (longitude defined in 0~360, converted to -180~180 internally) ---
REGIONS = {
    "Kuroshio":    {"lon_min": 120.0, "lon_max": 180.0, "lat_min": 5.0,  "lat_max": 45.0},
    "Gulf_Stream": {"lon_min": 275.0, "lon_max": 340.0, "lat_min": 10.0,   "lat_max": 42.0},
    "Brazil":      {"lon_min": 295.0, "lon_max": 320.0, "lat_min": -50.0, "lat_max": -25.0},
    "Agulhas":     {"lon_min": 5.0,   "lon_max": 55.0,  "lat_min": -50.0, "lat_max": -25.0},
}

'''REGIONS = {
   "Gulf_Stream": {"lon_min": 275.0, "lon_max": 345.0, "lat_min": 10.0, "lat_max": 42.0},
   "Gulf_Stream_2": {"lon_min": 275.0, "lon_max": 340.0, "lat_min": 10.0, "lat_max": 42.0},
   "Gulf_Stream_3": {"lon_min": 275.0, "lon_max": 335.0, "lat_min": 10.0, "lat_max": 42.0}
}'''

def lon_360_to_180(lon):
    return lon - 360.0 if lon > 180.0 else lon


# build the lon/lat grid (2040 lat, 4320 lon)
lat_step = 180.0 / 2040
lon_step = 360.0 / 4320
lat_vals = torch.linspace(-80 + 1 / 12, -80 + 2040 * 1 / 12, steps=2040).to(device)
lon_vals = torch.linspace(-180, -180 + (4320 - 1) * 1 / 12, steps=4320).to(device)

# build an ocean mask for each region (True = ocean points inside the region)
region_ocean_masks = {}
for name, r in REGIONS.items():
    lon_min, lon_max = lon_360_to_180(r["lon_min"]), lon_360_to_180(r["lon_max"])
    lat_min, lat_max = r["lat_min"], r["lat_max"]
    lat_sel = (lat_vals >= lat_min) & (lat_vals <= lat_max)
    lon_sel = (lon_vals >= lon_min) & (lon_vals <= lon_max)
    grid_sel = lat_sel[:, None] & lon_sel[None, :]  # [2040, 4320]
    region_ocean_masks[name] = (grid_sel & ~model.land_mask).to(device)

# Per-region MC: pass ~rmask as land_mask (True=exclude)
region_MC = {}
global_ocean_mask = ~model.land_mask  # [H, W] global ocean mask
for name, rmask in region_ocean_masks.items():  # ocean points within the target region, the True part of [2040,4320]
    # extract the indices of this region within the flattened ocean sequence
    r_flat_idx = rmask[global_ocean_mask]
    # slice the region's uv_clim and norm_w
    r_uv_clim = model.uv_clim[..., rmask]
    r_norm_w = model.norm_w[r_flat_idx]
    # re-normalize the latitude weights within the region
    r_norm_w = r_norm_w / r_norm_w.sum()

    # pass an all-False 1D land_mask to avoid shape conflicts inside MC
    r_land_mask = torch.zeros(r_flat_idx.sum().item(), dtype=torch.bool, device=device)
    region_MC[name] = MetricsCalculator(r_uv_clim, r_land_mask, r_norm_w, use_weight=True)

model.get_memory_usage()
model.eval()

dm = GlobalCurrentDataModule(config, data_dict['land_mask'])
dm.setup(stage='test')
test_dataset = dm.test_dataset
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)
print(f"Test loader length (num batches): {len(test_loader)}")
print(f"Test dataset length (num samples): {len(dm.test_dataset) if dm.test_dataset is not None else 'None'}")
# accumulators: per region
region_mse_sum = {name: torch.zeros(day_rank, nv).to(device) for name in REGIONS}
region_acc_sum = {name: torch.zeros(day_rank, nv).to(device) for name in REGIONS}
region_eke_mse_sum = {name: torch.zeros(day_rank).to(device) for name in REGIONS}
region_eke_acc_sum = {name: torch.zeros(day_rank).to(device) for name in REGIONS}

with torch.no_grad():
    total_samples = 0
    for batch in tqdm(test_loader, desc=f'Check testing-set ...', unit='batch'):
        t = batch['time'].to(device)
        seq = batch['input'].to(device)
        target = batch['target'].to(device)
        total_samples += seq.shape[0]

        for i in range(start_day - 1):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(t, seq)
            t, seq = iterate_input(t, seq, nv, pred.float(), sql)

        for eva_num in range(start_day, iterate_day + 1):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(t, seq)
            t, seq = iterate_input(t, seq, nv, pred.float(), sql)

            # pred_fp32 = pred.float()
            pred_fp32 = pred.float().cpu()  # must cpu() to save
            pred_fp16 = pred_fp32[:, -nv:].numpy().astype(np.float16)
            pred_uv = torch.from_numpy(pred_fp16).to(torch.float32).to(device)

            # ---- global metrics ----
            if eva_num == 1:
                true = target[:, -nv:]
            else:
                true = target[:, -nv * eva_num:-nv * (eva_num - 1)]
            
            true_valid_mask = torch.isfinite(true) & (true > -5)
            true[~true_valid_mask] = 0.0

            # ---- per-region metrics ----
            for name, rmask in region_ocean_masks.items():
                # use rmask to extract the flattened prediction and truth within the region
                r_flat_idx = rmask[global_ocean_mask]
                r_eke_clim = eke_clim[..., rmask]  # [region_ocean_points,]
                # pred_uv [BS, 2, 2040, 4320]
                rflat = pred_uv[..., rmask]  # [BS, 2, region_ocean_points,]
                rtrue = true[..., r_flat_idx]
                rmetric = region_MC[name].compute_all(rflat, rtrue, r_eke_clim)
                region_mse_sum[name][eva_num - start_day] += rmetric['mse'].sum(dim=0)
                region_acc_sum[name][eva_num - start_day] += rmetric['acc'].sum(dim=0)
                region_eke_mse_sum[name][eva_num - start_day] += rmetric['eke_mse'].sum(dim=0)
                region_eke_acc_sum[name][eva_num - start_day] += rmetric['eke_acc'].sum(dim=0)

            del pred_fp32, pred, true
            torch.cuda.empty_cache()

# ---- print per region ----
for name in REGIONS:
    print(f'======== {name} ========')
    for day in range(day_rank):
        print(f'Day {day + start_day}: '
              f'RMSE_U={torch.sqrt(region_mse_sum[name][day, 0] / total_samples):.6f}, '
              f'RMSE_V={torch.sqrt(region_mse_sum[name][day, 1] / total_samples):.6f}, '
              f'ACC_U={region_acc_sum[name][day, 0] / total_samples:.6f}, '
              f'ACC_V={region_acc_sum[name][day, 1] / total_samples:.6f}, '
              f'EKE_RMSE={torch.sqrt(region_eke_mse_sum[name][day] / total_samples):.6f},'
              f'EKE_ACC={region_eke_acc_sum[name][day] / total_samples:.6f},')

# ---- output CSV ----
save_dir = "./validate_plot"
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "region_evaluate.csv")

rows = []
for day in range(day_rank):
    for name in REGIONS:
        row = {'Model': args.title, 'Lead_Day': day + start_day, 'Total_Samples': total_samples, 'Region': name}
        row[f'{name}_RMSE_U'] = torch.sqrt(region_mse_sum[name][day, 0] / total_samples).item()
        row[f'{name}_RMSE_V'] = torch.sqrt(region_mse_sum[name][day, 1] / total_samples).item()
        row[f'{name}_ACC_U'] = (region_acc_sum[name][day, 0] / total_samples).item()
        row[f'{name}_ACC_V'] = (region_acc_sum[name][day, 1] / total_samples).item()
        row[f'{name}_EKE_RMSE'] = torch.sqrt(region_eke_mse_sum[name][day] / total_samples).item()
        row[f'{name}_EKE_ACC'] = (region_eke_acc_sum[name][day] / total_samples).item()
        rows.append(row)

df_results = pd.DataFrame(rows)

if os.path.exists(csv_path):
    df_results.to_csv(csv_path, index=False, float_format='%.6f', mode='a', header=False)
    print(f"\n[INFO] results appended to: {csv_path}")
else:
    df_results.to_csv(csv_path, index=False, float_format='%.6f', mode='w', header=True)
    print(f"\n[INFO] created new file and saved to: {csv_path}")
