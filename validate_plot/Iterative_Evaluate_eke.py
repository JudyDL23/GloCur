import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from model.config import Config
import os
import argparse
from data.preprocessor import load_and_process_data
from data.dataset import GlobalCurrentDataModule
from lightning_modules.lit_module import LitCurrentModule
from model.SwTEarth_cyclic import SwinTransformer
from torch.utils.data import DataLoader
from utils.visualization import *
from utils.physics import iterate_input
from utils.metrics import MetricsCalculator

# DataProcessor
# Note! the code itself stores the metrics for every day, so the corresponding first and third quartiles can be plotted!!
parser = argparse.ArgumentParser(description="Iterative Evaluate")
parser.add_argument('--loss_mode', type=str, default='1', help='Loss functions mode (e.g., 1, 2, 3)')
parser.add_argument('--l2_coeff', type=float, default=1.00, help='Coefficient for loss2 terms')
parser.add_argument('--l3_coeff', type=float, default=1.00, help='Coefficient for loss3 terms')
parser.add_argument('--l4_coeff', type=float, default=1.00, help='Coefficient for loss4 terms')
parser.add_argument('--title', type=str,
                    default="GloCur_2DCU_no_softmax_2to1_93_13_L123_1_5_0_RK4_twoway_0.5_0.5_25epoch_seed42",
                    help='Model Parameters Files')
parser.add_argument('--pfx', type=str, default="GloCur_L123_1_5_0", help='Model Parameters Files Prefix')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()
device = torch.device('cuda:0')

start_day = 1
iterate_day = 15
day_rank = iterate_day - start_day + 1
config = Config(args, train_shuffle=False)
config.output_len = iterate_day
config.dataset_start = "1993-01-01"
config.train_start = "1993-01-01"
config.val_start = "2014-01-01"
config.test_start = "2019-10-01"
config.test_end = "2020-01-01"
config.train_range = "93_13"
config.batch_size = 1  # 8 8:OOM!
nv = config.num_var
sql = config.seq_len

metric_cache = {}
suffixes = ["acc", "mse", "eke_mse", "eke_acc"]
for i in range(start_day, iterate_day + 1):
    for suffix in suffixes:
        metric_cache[f"{i}d_{suffix}"] = []

data_dict = load_and_process_data(config)
""" 
'land_mask', 'sla_clim', 'mean_std', 'seabed.to(device)', 'interval_x.to(device)', 'interval_y.to(device)', 
'coord_grid.to(device)', 'ind_grid.to(device)', 'device', 'bottom_lat', 'w_lat.to(device)', 'masked_w.to(device)'
'norm_w.to(device)'
"""
title = args.title
print(title)
eke_clim_pth = './source/Global_eke_clim_93_13.npy'
save_dir = "./validate_plot/"
eke_clim = torch.from_numpy(np.load(eke_clim_pth)).to(device)
if eke_clim.shape[-2] != 2040:
    eke_clim = eke_clim[1:, :]

if not os.path.exists(os.path.join(save_dir, f'plot_file/{args.pfx}_17_19_curve.pth')):
    model = SwinTransformer(data_dict, config, spemb=True, img_size=(2040, 4320), patch_size=12,
                            embed_dim=192, depths=(4,), bottle_neck=(4,), num_heads=(4, 8, 4),
                            window_size=10, mlp_ratio=4., ape=False, use_checkpoint=False)

    litmodel = LitCurrentModule.load_from_checkpoint(
        f'./ckpt/{title}.ckpt',
        map_location=device,
        model=model,  # pass any additional arguments if needed
        data_dict=data_dict,  # pass constructor arguments at load time
        config=config,
        strict=False
    )
    model = litmodel.model.to(device)
    model.eval()
    model.requires_grad_(False)

    MC = MetricsCalculator(model.uv_clim, model.land_mask, model.norm_w, use_weight=True)
    model.get_memory_usage()

    dm = GlobalCurrentDataModule(config, data_dict['land_mask'])  # dataset needs mask on cpu
    dm.setup(stage='test')  # key: initialize test_dataset
    test_dataset = dm.test_dataset
    test_loader = DataLoader(test_dataset,
                             batch_size=4,  # use the configured batch_size  batch_size=8: OOM!
                             num_workers=0,  # increase the number of parallel processes
                             pin_memory=False,  # speed up GPU transfer
                             shuffle=False)

    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Check testing-set ...', unit='batch'):
            t = batch['time'].to(device)
            seq = batch['input'].to(device)
            target = batch['target'].to(device)
            total_samples += seq.shape[0]

            for eva_num in range(start_day, iterate_day + 1):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = model(t, seq)
                t, seq = iterate_input(t, seq, nv, pred.float(), sql)

                pred_fp32 = pred.float()
                flat = pred_fp32[..., :nv, ~model.land_mask]  # batch size * num_v * 37617'''
                if eva_num == 1:
                    true = target[:, -nv:]  # batch size * num_v * 37617
                else:
                    true = target[:, -nv * eva_num:-nv * (eva_num - 1)]

                batch_metric = MC.compute_all(flat, true, eke_clim)  # [batch_size, 2]
                # append the current batch metrics to the cache list
                metric_cache[f"{eva_num}d_acc"].append(batch_metric["acc"].cpu())  # [b, 2]
                metric_cache[f"{eva_num}d_mse"].append(batch_metric["mse"].cpu())  # [b, 2]
                metric_cache[f"{eva_num}d_eke_mse"].append(batch_metric["eke_mse"].cpu())  # [b, 1]
                metric_cache[f"{eva_num}d_eke_acc"].append(batch_metric["eke_acc"].cpu())  # [b, 1]

        # ========== after the loop: concatenate all batches along the batch dimension ==========
        all_metrics = {}
        for key, tensor_list in metric_cache.items():
            if len(tensor_list) == 0:
                raise RuntimeError(f"{key} collected no batch metrics!")
            all_metrics[key] = torch.cat(tensor_list, dim=0)  # dim=0 concatenate along the batch dimension

        package = {
            "metrics": all_metrics,
            "total_samples": total_samples
        }
        torch.save(package, os.path.join(save_dir, f'plot_file/{args.pfx}_17_19_curve.pth'))

else:
    package = torch.load(os.path.join(save_dir, f'plot_file/{args.pfx}_17_19_curve.pth'), map_location="cpu")

loaded_metrics = package["metrics"]
total_samples = package["total_samples"]
metric_mean = {}
for key, val in loaded_metrics.items():
    metric_mean[key] = val.mean(dim=0)

for day in range(start_day, iterate_day + 1):
    print(f'Day {day}')
    print(f'rmse_u: {torch.sqrt(metric_mean[f"{day}d_mse"][0]).item():.6f}')
    print(f'rmse_v: {torch.sqrt(metric_mean[f"{day}d_mse"][1]).item():.6f}')
    print(f'acc_u: {metric_mean[f"{day}d_acc"][0].item():.6f}')
    print(f'acc_v: {metric_mean[f"{day}d_acc"][1].item():.6f}')
    print(f'eke_rmse: {torch.sqrt(metric_mean[f"{day}d_eke_mse"]).item():.6f}')
    print(f'eke_acc: {metric_mean[f"{day}d_eke_acc"].item():.6f}')
    print(f'total_samples={total_samples}\n')

# create a unified csv save directory
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "newlatw.csv")
# assemble the DataFrame
rows = []
for day in range(start_day, iterate_day + 1):
    row = {
        'Task': 'Iterative Evaluation (Including EKE)',
        'Model': title,
        'Day': day,
        'RMSE_U': torch.sqrt(metric_mean[f"{day}d_mse"][0]).item(),
        'RMSE_V': torch.sqrt(metric_mean[f"{day}d_mse"][1]).item(),
        'ACC_U': metric_mean[f"{day}d_acc"][0].item(),
        'ACC_V': metric_mean[f"{day}d_acc"][1].item(),
        'EKE_ACC': metric_mean[f"{day}d_eke_acc"].item(),
        'EKE_RMSE': torch.sqrt(metric_mean[f"{day}d_eke_mse"]).item(),
        'Total_Samples': total_samples
    }
    rows.append(row)
df_results = pd.DataFrame(rows)  # write row by row, all columns must have the same length

# write CSV (append without header if the file exists, otherwise create a new one)
if os.path.exists(csv_path):
    # exists: append mode, do not write the header
    df_results.to_csv(csv_path, index=False, float_format='%.6f', mode='a', header=False)
    print(f"\n[INFO] results appended to: {csv_path}")
else:
    # does not exist: create a new file, write the header
    df_results.to_csv(csv_path, index=False, float_format='%.6f', mode='w', header=True)
    print(f"\n[INFO] created new file and saved to: {csv_path}")
