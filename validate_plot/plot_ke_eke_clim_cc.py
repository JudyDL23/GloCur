import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from model.config import Config
import argparse
from data.preprocessor import load_and_process_data
from data.dataset import GlobalCurrentDataModule
from lightning_modules.lit_module import LitCurrentModule
from model.SwTEarth import SwinTransformer
from utils.visualization import *
from utils.physics import iterate_input
import matplotlib.pyplot as plt
from utils.metrics import MetricsCalculator


parser = argparse.ArgumentParser(description="KE EKE Clim")
parser.add_argument('--loss_mode', type=str, default='1', help='Loss functions mode (e.g., 1, 2, 3)')
parser.add_argument('--l2_coeff', type=float, default=1.00, help='Coefficient for loss2 terms')
parser.add_argument('--l3_coeff', type=float, default=1.00, help='Coefficient for loss3 terms')
parser.add_argument('--l4_coeff', type=float, default=1.00, help='Coefficient for loss4 terms')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--lead_day', type=int, default=1, help='Lead Dyas of Forecasting')
args = parser.parse_args()

device = torch.device('cuda:0')

start_day = 1
iterate_day = args.lead_day
day_rank = iterate_day - start_day + 1
config = Config(args, train_shuffle=False)
config.lead = iterate_day
config.dataset_start = "1993-01-01"
config.train_start = "1993-01-01"
config.val_start = "2014-01-01"
config.test_start = "2017-01-01"
config.test_end = "2020-01-01"
config.train_range = "93_13"
config.batch_size = 4
nv = config.num_var
sql = config.seq_len
data_dict = load_and_process_data(config)
DPI = 600

title_data = 'GloCur_2to1_93_13_L1_25epoch_seed42'
title_phy = 'GloCur_2DCU_no_softmax_2to1_93_13_L123_1_5_0_RK4_twoway_0.5_0.5_25epoch_seed42'
save_pth = './validate_plot/plot_file'
uv_clim_pth = './source/Global_uv_clim_93_13.npy'
fig_out_pth = './fig'
uv_clim = np.load(uv_clim_pth)
if uv_clim.shape[-2] != 2040:
    uv_clim = uv_clim[..., 1:, :]

model_data = SwinTransformer(data_dict, config, spemb=True, img_size=(2040, 4320), patch_size=12,
                        embed_dim=192, depths=(4,), bottle_neck=(4,), num_heads=(4, 8, 4),
                        window_size=10, mlp_ratio=4., ape=False, use_checkpoint=False).to(device)

model_phy = SwinTransformer(data_dict, config, spemb=True, img_size=(2040, 4320), patch_size=12,
                        embed_dim=192, depths=(4,), bottle_neck=(4,), num_heads=(4, 8, 4),
                        window_size=10, mlp_ratio=4., ape=False, use_checkpoint=False).to(device)
lm_gpu = model_data.land_mask
lm_cpu = lm_gpu.cpu().numpy()

model_state_data = torch.load(f'./params/{title_data}.pth',
                    map_location=device)
model_state_phy = torch.load(f'./params/{title_phy}.pth',
                    map_location=device)
model_data.load_state_dict(model_state_data, strict=False)
model_phy.load_state_dict(model_state_phy, strict=False)
model_data.eval()
model_phy.eval()

dm = GlobalCurrentDataModule(config, data_dict['land_mask'])
dm.setup(stage='test')
test_loader = dm.test_dataloader()

ke_data_base = torch.zeros(2040, 4320).to(device)
ke_phy_base  = torch.zeros(2040, 4320).to(device)
ke_label_base = torch.zeros(2040, 4320).to(device)

eke_data_base = torch.zeros(2040, 4320).to(device)
eke_phy_base  = torch.zeros(2040, 4320).to(device)
eke_label_base = torch.zeros(2040, 4320).to(device)

ke_data_file  = os.path.join(save_pth, f'pred_ke_{iterate_day}d_data_17_19_clim.npy')
ke_phy_file   = os.path.join(save_pth, f'pred_ke_{iterate_day}d_phy_17_19_clim.npy')
ke_label_file = os.path.join(save_pth, 'label_ke_17_19_clim.npy')

eke_data_file  = os.path.join(save_pth, f'pred_eke_{iterate_day}d_data_17_19_clim.npy')
eke_phy_file   = os.path.join(save_pth, f'pred_eke_{iterate_day}d_phy_17_19_clim.npy')
eke_label_file = os.path.join(save_pth, 'label_eke_17_19_clim.npy')

need_ke_label  = not os.path.exists(ke_label_file)
need_eke_label = not os.path.exists(eke_label_file)

if not (os.path.exists(ke_data_file) and os.path.exists(ke_phy_file) and
        os.path.exists(eke_data_file) and os.path.exists(eke_phy_file)):
    with torch.no_grad():
        total_samples = 0
        uv_clim_t = torch.from_numpy(uv_clim).float().to(device)
        for batch in tqdm(test_loader, desc=f'Check testing-set ...', unit='batch'):
            t_data = batch['time'].to(device)
            t_phy = t_data.clone()
            seq_data = batch['input'].to(device)
            seq_phy = seq_data.clone()
            total_samples += seq_data.shape[0]

            if need_ke_label or need_eke_label:
                target = batch['target'].to(device)
                label_rec = torch.zeros(seq_data.shape[0], 2, 2040, 4320).to(device)
                label_rec[..., ~lm_gpu] = target

            for eva_num in range(start_day, iterate_day + 1):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred_data = model_data(t_data, seq_data)
                    pred_phy = model_phy(t_phy, seq_phy)

                t_data, seq_data = iterate_input(t_data, seq_data, nv, pred_data.float(), sql)
                t_phy, seq_phy = iterate_input(t_phy, seq_phy, nv, pred_phy.float(), sql)

            pred_data32 = pred_data.float()
            pred_phy32 = pred_phy.float()

            ke_data = 0.5 * torch.sum(pred_data32 ** 2, dim=1)
            ke_phy  = 0.5 * torch.sum(pred_phy32 ** 2, dim=1)
            ke_data_base += torch.sum(ke_data, dim=0)
            ke_phy_base  += torch.sum(ke_phy, dim=0)

            delta_data = pred_data32 - uv_clim_t.unsqueeze(0)
            delta_phy  = pred_phy32 - uv_clim_t.unsqueeze(0)
            eke_data = 0.5 * torch.sum(delta_data ** 2, dim=1)
            eke_phy  = 0.5 * torch.sum(delta_phy ** 2, dim=1)
            eke_data_base += torch.sum(eke_data, dim=0)
            eke_phy_base  += torch.sum(eke_phy, dim=0)

            if need_ke_label or need_eke_label:
                ke_label = 0.5 * torch.sum(label_rec ** 2, dim=1)
                delta_label = label_rec - uv_clim_t.unsqueeze(0)
                eke_label = 0.5 * torch.sum(delta_label ** 2, dim=1)
                ke_label_base  += torch.sum(ke_label, dim=0)
                eke_label_base += torch.sum(eke_label, dim=0)

        np.save(ke_phy_file,  (ke_phy_base  / total_samples).cpu().numpy())
        np.save(ke_data_file, (ke_data_base / total_samples).cpu().numpy())
        np.save(eke_phy_file,  (eke_phy_base  / total_samples).cpu().numpy())
        np.save(eke_data_file, (eke_data_base / total_samples).cpu().numpy())

        if need_ke_label:
            np.save(ke_label_file, (ke_label_base / total_samples).cpu().numpy())
        if need_eke_label:
            np.save(eke_label_file, (eke_label_base / total_samples).cpu().numpy())
        print(f'Forecast day {eva_num} days saved.')

# ==================== latitude-weighted PCC ====================
n_day = 15
dodgerblue = '#1590FF'
tomato = '#FF6347'

# build latitude weights consistent with the data grid (1/12°, -80+1/12 ~ 90, 2040 rows total)
lat_1d = -80 + 1/12 + np.arange(2040) * (1/12)
w_lat_2d = np.cos(np.deg2rad(lat_1d))[:, None] * np.ones((1, 4320), dtype=np.float32)
w_ocean = w_lat_2d[~lm_cpu].ravel().astype(np.float64)


def weighted_pcc(x, y, w):
    """Latitude-weighted Pearson correlation coefficient. x, y, w are all 1D arrays."""
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    w = w.ravel().astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[m], y[m], w[m]
    W = w.sum()
    mx = (w * x).sum() / W
    my = (w * y).sum() / W
    dx = x - mx
    dy = y - my
    cov = (w * dx * dy).sum() / W
    vx = (w * dx * dx).sum() / W
    vy = (w * dy * dy).sum() / W
    return cov / (np.sqrt(vx * vy) + 1e-12)


def check_all_exist(prefix):
    return all(
        os.path.exists(os.path.join(save_pth, f"pred_{prefix}_{day}d_phy_17_19_clim.npy"))
        and os.path.exists(os.path.join(save_pth, f"pred_{prefix}_{day}d_data_17_19_clim.npy"))
        for day in range(1, n_day + 1)
    )


ke_all_exist  = check_all_exist('ke')  and os.path.exists(ke_label_file)
eke_all_exist = check_all_exist('eke') and os.path.exists(eke_label_file)

if ke_all_exist and eke_all_exist:
    print('All exist!')
    ke_label_clim  = np.load(ke_label_file)
    eke_label_clim = np.load(eke_label_file)

    ke_curve  = np.zeros((2, n_day))
    eke_curve = np.zeros((2, n_day))

    for i in range(n_day):
        # KE (latitude-weighted)
        ke_phy_day  = np.load(os.path.join(save_pth, f'pred_ke_{i+1}d_phy_17_19_clim.npy'))
        ke_data_day = np.load(os.path.join(save_pth, f'pred_ke_{i+1}d_data_17_19_clim.npy'))
        ke_curve[0, i] = weighted_pcc(ke_data_day[~lm_cpu], ke_label_clim[~lm_cpu], w_ocean)
        ke_curve[1, i] = weighted_pcc(ke_phy_day[~lm_cpu],  ke_label_clim[~lm_cpu], w_ocean)

        # EKE (latitude-weighted)
        eke_phy_day  = np.load(os.path.join(save_pth, f'pred_eke_{i+1}d_phy_17_19_clim.npy'))
        eke_data_day = np.load(os.path.join(save_pth, f'pred_eke_{i+1}d_data_17_19_clim.npy'))
        eke_curve[0, i] = weighted_pcc(eke_data_day[~lm_cpu], eke_label_clim[~lm_cpu], w_ocean)
        eke_curve[1, i] = weighted_pcc(eke_phy_day[~lm_cpu],  eke_label_clim[~lm_cpu], w_ocean)

    # ---------- 1x2 subplots ----------
    x = np.arange(1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=DPI)

    panel_cfg = [
        (axes[0], ke_curve,  'KE PCC'),
        (axes[1], eke_curve, 'EKE PCC'),
    ]

    legend_handles = None
    legend_labels = None

    for ax, curve, title in panel_cfg:
        line_data, = ax.plot(x, curve[0, :10], linestyle='--', linewidth=3, color=tomato, alpha=1)
        line_phy,  = ax.plot(x, curve[1, :10], linestyle='-',  linewidth=3, color=dodgerblue, alpha=1)
        if legend_handles is None:
            legend_handles = [line_data, line_phy]
            legend_labels = ['Data-Driven', 'Physics-Guided']

        ax.grid(True, color='gray', linestyle='-', linewidth=0.2)
        ax.set_xlabel('Lead Time (days)', fontsize=18)
        ax.set_ylabel('PCC', fontsize=17)
        ax.set_title(title, fontsize=18)
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xticklabels(['1', '4', '7', '10'], fontsize=17)
        ax.set_xlim(1, 10)

        data_min = np.min(curve[:, :10])
        data_max = np.max(curve[:, :10])
        rank = data_max - data_min
        ax.set_ylim(data_min - 0.1 * rank, data_max + 0.1 * rank)
        ax.tick_params(axis='y', labelsize=15)

        y_ref = curve[1, 9]
        print(f'{title} data-driven:{curve[0, 9]}')
        ax.axhline(y=y_ref, linestyle='-.', color='black', linewidth=3)
        y_frac = (y_ref - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(0.1, y_frac + 0.04,
                f'PCC = {y_ref:.3f}',
                transform=ax.transAxes,
                fontsize=17, fontweight='bold')

    fig.legend(handles=legend_handles, labels=legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=2, fontsize=17, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    filename = f'Clim Weighted PCC KE and EKE 1-10_2017-2019.png'
    plt.savefig(os.path.join(fig_out_pth, filename), bbox_inches='tight')
    print('save successfully!!')
    plt.close()
else:
    print('Somthing still does not exist!')