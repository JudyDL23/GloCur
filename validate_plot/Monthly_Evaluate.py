import pandas as pd
from tqdm import tqdm
from model.config import Config
import argparse
from data.preprocessor import load_and_process_data
from data.dataset import GlobalCurrentDataModule
from model.SwTEarth import SwinTransformer
from utils.visualization import *
from torch.utils.data import Dataset, DataLoader
from utils.physics import iterate_input
from utils.metrics import MetricsCalculator


# DataProcessor
parser = argparse.ArgumentParser(description="Global SLA Predictor Training Config")
parser.add_argument('--loss_mode', type=str, default='1', help='Loss functions mode (e.g., 1, 2, 3)')
parser.add_argument('--l2_coeff', type=float, default=1.0, help='Coefficient for loss2 terms')
parser.add_argument('--l3_coeff', type=float, default=1.0, help='Coefficient for loss3 terms')
parser.add_argument('--l4_coeff', type=float, default=1.0, help='Coefficient for loss4 terms')
parser.add_argument('--title', type=str, default="GloCur_2DCU_no_softmax_2to1_93_13_L123_1_5_0_RK4_twoway_0.5_0.5_25epoch_seed42", help='ckpt title')
parser.add_argument('--test_year', type=int, default=2017, help='test_year')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

device = torch.device('cuda:0')

iterate_day = 10
config = Config(args, train_shuffle=False)
config.lead = iterate_day  # keep the output_len unchanged, equals fcst_len
config.dataset_start = "1993-01-01"
config.train_range = '93_13'
config.batch_size = 4
nv = config.num_var
sql = config.seq_len

mse_sum = torch.zeros(12, nv).to(device)
acc_sum = torch.zeros(12, nv).to(device)
eke_mse_sum = torch.zeros(12).to(device)
eke_acc_sum = torch.zeros(12).to(device)
total_samples = torch.zeros(12).to(device)
data_dict = load_and_process_data(config)

eke_clim_pth = './source/Global_eke_clim_93_13.npy'
save_dir = "./validate_plot/"
eke_clim = torch.from_numpy(np.load(eke_clim_pth)).to(device)
if eke_clim.shape[-2] != 2040:
    eke_clim = eke_clim[1:, :]
""" 
'land_mask', 'sla_clim', 'mean_std', 'seabed.to(device)', 'interval_x.to(device)', 'interval_y.to(device)', 
'coord_grid.to(device)', 'ind_grid.to(device)', 'device', 'bottom_lat', 'w_lat.to(device)', 'masked_w.to(device)'
'norm_w.to(device)'
"""
# 'GloCur_2to1_93_13_L1_25epoch_seed42'
# 'GloCur_2to1_93_17_L123_1_5_0_RK4_twoway_0.5_0.5_25epoch_seed42'
model = SwinTransformer(data_dict, config, spemb=True, img_size=(2040, 4320), patch_size=12,
                        embed_dim=192, depths=(4,), bottle_neck=(4,), num_heads=(4, 8, 4),
                        window_size=10, mlp_ratio=4., ape=False, use_checkpoint=False).to(device)
# embed_dim=192, ps = 12, ws=10, depths=(4), bottle_neck=(4,), num_heads=(4, 8, 4) 11,457,355
MC = MetricsCalculator(model.uv_clim, model.land_mask, model.norm_w, use_weight=True)
model.get_memory_usage()
print(args.title)
print(args.test_year)
model_state_dict = torch.load(f'./params/{args.title}.pth',
                                      map_location=device)
model.load_state_dict(model_state_dict, strict=False)
model.eval()
# model_state_dict = torch.load(f'./test/{config.title}.pth', map_location=device)
# model.load_state_dict(model_state_dict)
test_year = args.test_year
title = args.title
with torch.no_grad():
    for mon in range(1, 13):
        if mon == 1:
            config.train_start = f"{test_year - 1}-12-30"
            config.val_start = f"{test_year - 1}-12-31"
            config.test_start = f"{test_year}-{mon:02d}-01"
            config.test_end = f"{test_year}-{mon + 1:02d}-01"
        elif mon < 12:
            config.train_start = f"{test_year}-{mon - 1:02d}-27"
            config.val_start = f"{test_year}-{mon - 1:02d}-28"
            config.test_start = f"{test_year}-{mon:02d}-01"
            config.test_end = f"{test_year}-{mon + 1:02d}-01"
        elif mon == 12:
            config.train_start = f"{test_year}-11-29"
            config.val_start = f"{test_year}-11-30"
            config.test_start = f"{test_year}-{mon:02d}-01"
            config.test_end = f"{test_year + 1}-01-01"

        dm = GlobalCurrentDataModule(config, data_dict['land_mask'])  # dataset needs mask on cpu
        dm.setup(stage='test')
        test_dataset = dm.test_dataset
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        for batch in tqdm(test_loader, desc=f'Check testing-set ...', unit='batch'):
            t = batch['time'].to(device)
            seq = batch['input'].to(device)
            target = batch['target'].to(device)
            total_samples[mon - 1] += seq.shape[0]
            # seq0 = seq
            for i in range(iterate_day):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = model(t, seq)
                t, seq = iterate_input(t, seq, nv, pred.float(), sql)

            pred_fp32 = pred.float()
            # flat = seq0[..., :n_var, ~land_mask]  # batch size * num_v * 37617
            flat = pred_fp32[..., ~model.land_mask]  # batch size * num_v * 37617
            true = target[:, -nv:]  # batch size * num_v * 37617
            # batch size * 2 * 37617
            batch_metric = MC.compute_all(flat, true, eke_clim)  # [batch_size, 2]
            mse_sum[mon - 1] += batch_metric['mse'].sum(dim=0)  # [2,]
            acc_sum[mon - 1] += batch_metric['acc'].sum(dim=0)  # [2,]
            eke_mse_sum[mon - 1] += batch_metric['eke_mse'].sum(dim=0)  # [1,]
            eke_acc_sum[mon - 1] += batch_metric['eke_acc'].sum(dim=0)  # [1,]
            # bs * 2 -> 1 * 2
            # persistence

for mon in range(12):
    print(f'Month {mon + 1}')
    print(f'rmse_u: {torch.sqrt(mse_sum[mon, 0] / total_samples[mon]):.6f}')
    print(f'rmse_v: {torch.sqrt(mse_sum[mon, 1] / total_samples[mon]):.6f}')
    print(f'acc_u: {acc_sum[mon, 0] / total_samples[mon]:.6f}')
    print(f'acc_v: {acc_sum[mon, 1] / total_samples[mon]:.6f}')
    print(f'eke_rmse: {torch.sqrt(eke_mse_sum[mon] / total_samples[mon]):.6f}')
    print(f'eke_acc: {eke_acc_sum[mon] / total_samples[mon]:.6f}')
    print(f'total_samples={total_samples[mon]}\n')

# create a unified csv save directory
save_dir = "./validate_plot/"
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "newlatw.csv")

# assemble the DataFrame
rows = []
for mon in range(12):
    row = {
        'Year': test_year,
        'Task': 'Iterative Evaluation Monthly',
        'Model': title,
        'Month': mon + 1,
        'RMSE_U': torch.sqrt(mse_sum[mon, 0] / total_samples[mon]).cpu().item(),
        'RMSE_V': torch.sqrt(mse_sum[mon, 1] / total_samples[mon]).cpu().item(),
        'ACC_U': (acc_sum[mon, 0] / total_samples[mon]).cpu().item(),
        'ACC_V': (acc_sum[mon, 1] / total_samples[mon]).cpu().item(),
        'EKE_RMSE': torch.sqrt(eke_mse_sum[mon] / total_samples[mon]).cpu().item(),
        'EKE_ACC': (eke_acc_sum[mon] / total_samples[mon]).cpu().item(),
        'Total_Samples': total_samples[mon].cpu().item()
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


