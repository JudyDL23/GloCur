import torch
import numpy as np
from tqdm import tqdm
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from data.preprocessor import load_and_process_data
from model.config import Config
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.dataset import GlobalCurrentDataModule
from model.LossFunc import *
from model.SwTEarth import SwinTransformer
from validate_plot.plot_variable import create_enhanced_norm
import argparse
torch.autograd.set_detect_anomaly(True)


# fw: forward
# bk: backward
def check_surrounding(ocean_binary, mask, buffer):
    ocean_binary = global_expand(ocean_binary, buffer)
    mask_exp = global_expand(mask, buffer)
    k_minus = torch.zeros((2 * buffer + 1) ** 2 - 1, 1, 2 * buffer + 1, 2 * buffer + 1, device=mask.device)
    kernels = torch.zeros((2 * buffer + 1) ** 2, 1, 2 * buffer + 1, 2 * buffer + 1, device=mask.device)
    count = torch.ones(1, 1, 2 * buffer + 1, 2 * buffer + 1, device=mask.device)
    positions = [(i, j) for i in range(2 * buffer + 1) for j in range(2 * buffer + 1) if not (i == buffer and j == buffer)]
    for idx, (i, j) in enumerate(positions):
        k_minus[idx, 0, buffer, buffer] = -1
        k_minus[idx, 0, i, j] = 1  # k_minux for calc Δq
        kernels[idx, 0, i, j] = 1
    kernels[-1, 0, buffer, buffer] = 1  # (buffer*2+1)^2 * 1 * k * k
    # get the effective ocean area
    valid_mask = (F.conv2d(ocean_binary, k_minus, padding=buffer) == 0).float() * ocean_binary
    valid_mask = torch.cat([valid_mask, ocean_binary], dim=1)  # calc Δq at its own location
    # 1 * [(buffer*2 + 1)^2] * H * W  #

    # the tested point and the host point are both in ocean
    o_num = F.conv2d(ocean_binary, count, padding=buffer) * ocean_binary + mask_exp.float()
    # including the center, how many sea gird in the ambient area
    # [1 * 48 * H * W ] * [1, 1, H, W] = [1 * 48 * H * W ]
    return kernels, \
           valid_mask[..., :-buffer, buffer:-buffer], \
           o_num[..., :-buffer, buffer:-buffer]
  
  
def calculate_pearson(x_list, y_list):
    """
    Compute the Pearson correlation coefficient of two lists.
        x_list, y_list: two equal-length numeric lists
        r: correlation coefficient, range [-1, 1]
    """
    x = np.array(x_list)
    y = np.array(y_list)
    x_mean = x - np.mean(x)
    y_mean = y - np.mean(y)

    # Compute the correlation coefficient
    numerator = np.sum(x_mean * y_mean)
    denominator = np.sqrt(np.sum(x_mean**2) * np.sum(y_mean**2))

    if denominator == 0:
        return 0  # return 0 when there is no variation
    return numerator / denominator


class FieldDeltaPV(nn.Module):
    def __init__(self, model, resolution, exp_size):
        super(FieldDeltaPV, self).__init__()
        self.getIntp = GetIntp_parallel(model, resolution, exp_size)
        self.register_buffer('lat', model.coor[1], persistent=False)
        self.register_buffer('mask', model.land_mask, persistent=False)
        self.register_buffer('iy', model.iy, persistent=False)
        self.res = resolution

    def forward(self, interpolated_field, grid_field, motion):
        # y_degree = (motion[:, 1:2] / self.iy) * self.res
        # f_itpl = 2 * 7.292e-5 * torch.sin((self.lat + y_degree) * torch.pi / 180)
        # f_grid = 2 * 7.292e-5 * torch.sin(self.lat * torch.pi / 180)
        interpolating_field = self.getIntp(interpolated_field, motion)
        
        # q_itpl = interpolating_field + f_itpl
        # q_grid = grid_field + f_grid

        q_itpl = interpolating_field
        q_grid = grid_field
        delta_q = q_itpl - q_grid
        return delta_q, delta_q[..., ~self.mask]


class ValiSigmaQ(nn.Module):  # batchsize * (num variables * seq length) * latitude* longitude 6*193*505
    def __init__(self, model, resolution, exp_size, thresholds, names):
        super(ValiSigmaQ, self).__init__()
        self.register_buffer('mask', model.land_mask, persistent=False)
        self.register_buffer('ix', model.ix, persistent=False)
        self.register_buffer('iy', model.iy, persistent=False)
        self.register_buffer('f', 2 * 7.292e-05 * torch.sin(torch.pi * model.coor[1] / 180), persistent=False)
        self.register_buffer('coor', model.coor, persistent=False)
        self.register_buffer('lat', model.coor[1], persistent=False)
        self.register_buffer('norm_w', model.norm_w, persistent=False)
        self.register_buffer('clim', model.uv_clim[..., ~self.mask], persistent=False)
        self.dp = FieldDeltaPV(model, resolution, exp_size)
        self.w_lat = torch.cos(self.lat * torch.pi / 180)
        self.lat_weights = self.w_lat[~self.mask]
        self.res = resolution
        self.eps = exp_size

        self.thresholds = thresholds
        self.names = names

        self.RK1 = RK1(model)
        self.RK2 = RK2(model, self.res, self.eps, 'parallel')
        self.RK3 = RK3(model, self.res, self.eps, 'parallel')
        self.RK4 = RK4(model, self.res, self.eps, 'parallel')
        
        self.Pearson120 = LocalPearsonLoss(self.mask, 120, 60, 2000)  # 5°60 30 600  3°36 18 200
        self.Pearson60 = LocalPearsonLoss(self.mask, 60, 30, 600)  # 5°60 30 600  3°36 18 200
        self.Pearson36 = LocalPearsonLoss(self.mask, 36, 18, 300)  # 5°60 30 600  3°36 18 200

        ocean_binary = (~self.mask > 0.5).float().unsqueeze(0).unsqueeze(0)  # 1 * 1 * H * W
        # self.k3, self.valid_mask3, self.o_num3 = check_surrounding(ocean_binary, self.mask, 3)
        self.k2, self.valid_mask2, self.o_num2 = check_surrounding(ocean_binary, self.mask, 2)
        self.k1, self.valid_mask1, self.o_num1 = check_surrounding(ocean_binary, self.mask, 1)

    def calc_relu_sigma_dq(self, sigma, dq):
        # r8 = relu8[..., ~self.mask] # B,1,(~mask)
        # r8_ratio = ((r8 == 0).float() * self.lat_weights.view(1, 1, -1)).sum(dim=2).mean() / self.lat_weights.sum()
        abs_sigma = torch.abs(sigma)
        abs_dq = torch.abs(dq)
        ratios_dict = {}
        for thresh, name in zip(self.thresholds, self.names):  # area ratio
            relu_val = F.relu(abs_sigma - abs_dq * thresh)  # calc relu
            r = relu_val[..., ~self.mask]
            zero_mask = (r == 0).float()
            weighted_sum = (zero_mask * self.lat_weights.view(1, 1, -1)).sum(dim=2)
            weighted_sum = weighted_sum.squeeze(-1)

            ratio = weighted_sum / self.lat_weights.sum()
            ratios_dict[f'{name}_ratio'] = ratio.cpu().tolist()
        return ratios_dict
      
    def calc_sigma(self, last_sigma, present_sigma):
        # r8 = relu8[..., ~self.mask] # B,1,(~mask)
        # r8_ratio = ((r8 == 0).float() * self.lat_weights.view(1, 1, -1)).sum(dim=2).mean() / self.lat_weights.sum()
        sigma_dict = {}
        l_sigma = torch.abs(last_sigma)
        p_sigma = torch.abs(present_sigma)
        l_flat = l_sigma[..., ~self.mask] ** 2 * self.norm_w
        p_flat = p_sigma[..., ~self.mask] ** 2 * self.norm_w
        sigma_dict['last_sigma'] = torch.sum(l_flat, dim=-1).cpu().numpy()
        sigma_dict['present_sigma'] = torch.sum(p_flat, dim=-1).cpu().numpy()
        return sigma_dict

    def calc_dq_RK1_4(self, field, vel, time, previous_motion, interpolated_field, grided_field):
        dq_results = {}
        var = 'bk' if time < 0 else 'fw'
        for i in range(1, 5):
            rk_method = getattr(self, f'RK{i}')  # dynamic adjustment
            motion = rk_method(field, vel, time, previous_motion)  # (km), B, 2, H, W
            dq, dq_flat = self.dp(interpolated_field, grided_field, motion)
            dq_flat = dq_flat ** 2 * self.norm_w  # B, 1, (~mask)
            dq_results[f'{var}_q{i}'] = torch.sum(dq_flat, dim=-1).cpu().numpy()
            if i == 4:
                abs_motion = torch.sqrt(torch.sum(motion ** 2, dim=-3, keepdim=True)) # [bs, 1, H, W] retains dim1 for later processing
                dq_results[f'{var}_RK{i}_motion'] = abs_motion.detach().cpu().numpy()
                dq_RK4 = dq
        return dq_results, dq_RK4

    def calc_dq_mean(self, num, ambient_field, center_q, var, min=False):
        mean_q_results = {}
        for i in range(1, num + 1):
            # m = n * num
            kernel = getattr(self, f'k{i}')
            valid_mask = getattr(self, f'valid_mask{i}')
            ocean_num = getattr(self, f'o_num{i}')
            ambient_field_exp = global_expand(ambient_field, i)
            ambient_q = F.conv2d(ambient_field_exp, kernel, padding=i)
            dq = (ambient_q[..., :-i, i:-i] - center_q)
            dq = torch.abs(dq * valid_mask)  # 1,1,H,W --- (m^2) * 1 * m * m (conv) ---> 1 * (m^2) * H * W
            mean_dq = torch.sum(dq, dim=1, keepdim=True) / ocean_num
            mean_dq = mean_dq[..., ~self.mask] ** 2 * self.norm_w  # B, 1, (~mask)
            mean_q_results[f'{var}_mean_q{i}'] = torch.sum(mean_dq, dim=-1).cpu().numpy()
            if min & i == 1:
                dq_00 = dq + (1. - valid_mask)
                min_dq, min_ind = torch.min(dq_00, dim=1, keepdim=True)  # min_dq, min_id
                min_dq[..., self.mask] = 0
                # mean_q_results[f'{var}_min_dq'] = torch.sum(min_dq[..., ~self.mask] ** 2 * self.norm_w, dim=-1).cpu().numpy()
                mean_q_results[f'{var}_min_id'] = min_ind.detach().cpu().numpy()
                mean_q_results[f'{var}_min_dq_field'] = min_dq.detach().cpu().numpy()
        return mean_q_results

    def forward(self, data):  # in * 193 * 505
        pearson_dict = {}
        sigma_dq_plot = {}
        dt = 86400
        uv_last = data[:, 2:4, :, :]
        uv_now = data[:, :2, :, :]
        uv_last[..., self.mask] = 0  # B, 2, H, W
        uv_now[..., self.mask] = 0
        last_field = uv_last.clone()
        now_field = uv_now.clone()

        # calc partial
        u_x, u_y, v_x, v_y = calc_grad_4order(uv_last, self.ix, self.iy, self.eps)  # batch_size * 1 * lat * lon
        curl_last = (v_x - u_y) * 0.001
        sigma_last = torch.sqrt((u_x - v_y) ** 2 + (v_x + u_y) ** 2) * 0.001  # B,1,H,W
        u_x, u_y, v_x, v_y = calc_grad_4order(uv_now, self.ix, self.iy, self.eps)  # batch_size * 1 * lat * lon
        curl_now = (v_x - u_y) * 0.001
        sigma_now = torch.sqrt((u_x - v_y) ** 2 + (v_x + u_y) ** 2) * 0.001  # B,1,H,W

        # mean, max, min PV
        q_last_field = curl_last + self.f
        q_now_field = curl_now + self.f  # B, 1, H, W

        # Back-ward dp(interpolated_field, grid_field, motion)
        if True: 
            ambient_num = 2
            # now the field was transported from the last probable state
            bk_dq_dict, bk_dq_field = self.calc_dq_RK1_4(last_field, uv_last, -dt, torch.zeros_like(uv_last), q_last_field, q_now_field)  # !!!! curl_last, curl_now
            bk_mean_dq_dict = self.calc_dq_mean(ambient_num, q_last_field, q_now_field, 'bk', min=True)
            # num, ambient_field, center_field
            # the last grided field will transport to the future probable state
            fw_dq_dict, fw_dq_field = self.calc_dq_RK1_4(now_field, uv_now, dt, torch.zeros_like(uv_now), q_now_field, q_last_field)  # !!!! curl_now, curl_last
            fw_mean_dq_dict = self.calc_dq_mean(ambient_num, q_now_field, q_last_field, 'fw', min=True)

        # sigma Δq
        if True:
        		ratio_dict = self.calc_relu_sigma_dq(sigma_now + sigma_last, bk_dq_field.abs() + fw_dq_field.abs())
        
        # sigma
        if True:
        		sigma_dict = self.calc_sigma(sigma_last, sigma_now)
        
        # plot sigma dq
        if True:
            sigma_dq_plot['sum_sigma'] = (sigma_now + sigma_last).cpu().numpy()
            sigma_dq_plot['sum_dq'] = (bk_dq_field.abs() + fw_dq_field.abs()).cpu().numpy()
            sigma_dq_plot['lamda'] = ((sigma_now + sigma_last) / (bk_dq_field.abs() + fw_dq_field.abs() + 1e-8)).cpu().numpy()
            
        if True:
            loss120 = self.Pearson120(sigma_now + sigma_last, bk_dq_field.abs() + fw_dq_field.abs())
            loss60 = self.Pearson60(sigma_now + sigma_last, bk_dq_field.abs() + fw_dq_field.abs())
            loss36 = self.Pearson36(sigma_now + sigma_last, bk_dq_field.abs() + fw_dq_field.abs())
            pearson_dict['pearson120'] = loss120.cpu().numpy()
            pearson_dict['pearson60'] = loss60.cpu().numpy()
            pearson_dict['pearson36'] = loss36.cpu().numpy()

        # num, ambient_field, center_field
        all_dics = {**ratio_dict, **bk_dq_dict, **bk_mean_dq_dict, **fw_dq_dict, **fw_mean_dq_dict, **pearson_dict, **sigma_dq_plot, **sigma_dict}
        # all_dics = {**ratio_dict, **pearson_dict}

        return all_dics
        # ratio_dict: r1_ratio, r2_5_ratio, r5_ratio, r8_ratio, r10_ratio, r12_ratio, r15_ratio
        # bk_dq_dict: bk_q1, bk_q2, bk_q3, bk_q4, bk_RK4_motion, 
        # fw_dq_dict: fw_q1, fw_q2, fw_q3, fw_q4, fw_RK4_motion
        # bk_mean_dq_dict: bk_mean_q1, bk_mean_q2, bk_min_id, bk_min_dq_field
        # fw_mean_dq_dict: fw_mean_q1, fw_mean_q2, fw_min_id, fw_min_dq_field
        # sigma_dq_plot: sum_sigma, sum_dq
        # sigma_dict: last_sigma, present_sigma
        # pearson_dict: pearson120, pearson60, pearson36


class Plot2D:
    def __init__(self, coor, mask, ana_start, ana_end, positions, labels):
        self.coor = coor
        self.land = mask
        row = 2040  # 193
        col = 4320  # 505
        bottom_lat = -80 + 1/12  # -80 + 1/12
        bottom_lon = -180  # -180
        start_lat = -80 + 1/12  # -80 + 1/12
        start_lon = -180  # -180
        self.lon_rank = slice(int(12 * (start_lon - bottom_lon)), int(12 * (start_lon - bottom_lon) + col))
        self.lat_rank = slice(int(12 * (start_lat - bottom_lat)), int(12 * (start_lat - bottom_lat) + row))
        self.x = self.coor[0, self.lat_rank, self.lon_rank].cpu().numpy()  # longitude
        self.y = self.coor[1, self.lat_rank, self.lon_rank].cpu().numpy()  # latitude

        self.start = ana_start
        self.end = (datetime.strptime(ana_end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        self.all_day = (datetime.strptime(ana_end, "%Y-%m-%d") - datetime.strptime(ana_start, "%Y-%m-%d")).days
        self.midnum = all_day // 2
        self.mid_date = (datetime.strptime(ana_start, "%Y-%m-%d") + timedelta(days=self.midnum)).strftime("%Y-%m-%d")

        self.pos = positions  # x-axis positions  , 2.4, 3
        self.labels = labels
        
        self.colors = ['k', 'r', 'b', 'orange', 'purple', 'brown', 'pink', 'gray', 'g', 'cyan']

    def plot_subplots(self, sub_var, sub_name, unit, fig_name, cmp, draw_contour=False, norm_manual=False):
        """
        RK4_motion: list of length 9, each element is numpy array of shape (2, 193, 505)
        """
        fig, axes = plt.subplots(3, 3, figsize=(13, 7), subplot_kw={'projection': ccrs.PlateCarree()})
        axes_flat = axes.flatten()
        all_data = []
        for i in range(9):
            data = sub_var[i][0]  # data: 1, H, W
            all_data.append(data[~self.land])
        all_data_flat = np.concatenate(all_data)
        norm_var = create_enhanced_norm(all_data_flat, symmetric=False, enhance_factor=1)
        # vmin, vmax = np.percentile(all_data_flat, [0, 100])

        for i in range(9):
            ax = axes_flat[i]
            data = sub_var[i][0]
            str_name = sub_name[i]

            im = ax.pcolormesh(self.x, self.y, data,
                               cmap=cmp,
                               transform=ccrs.PlateCarree(),
                               shading='auto',
                               norm=Normalize(vmin=0, vmax=10) if norm_manual else norm_var,
                               zorder=1)
            
            if draw_contour:
                levels = [2, 4, 6, 8, 10]
                contour = ax.contour(self.x, self.y, data,
                                     levels=levels,
                                     colors='k',        # contour line color: black
                                     linewidths=0.5,     # line width
                                     alpha=1,          # transparency
                                     transform=ccrs.PlateCarree(),
                                     zorder=4)
                ax.clabel(contour, inline=True, fontsize=3, fmt='%.1f', zorder=5)

            ax.coastlines(resolution='50m', linewidth=0.4, zorder=3)
            ax.add_feature(cfeature.LAND, color='lightgray', zorder=2)
            ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=0)

            ax.set_title(str_name, fontsize=11)  # fontweight='bold'

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0)

        cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.75])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(0.5)
        cbar.set_label(unit, fontsize=11)

        # fig.suptitle(fig_name, fontsize=16, y=0.98)  # fontweight='bold',
        plt.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.03,
                            wspace=0.05, hspace=0.05)
        plt.savefig(f'./fig/fig2/'
                    f'{fig_name} {self.start}-{self.end} 600dpi.png', dpi=500, bbox_inches='tight', transparent=True, pil_kwargs={'optimize': True})
        # plt.show()
        plt.close(fig)

    def plot_multi_curves(self, collector, var_name, title):
        t = range(collector.n_group)
        all_values = []
        for suffix in ['_q4', '_q3', '_q2', '_q1', '_mean_q2']:  # ['_q4', '_q3', '_q2', '_q1', '_mean_q1', '_mean_q2']
            all_values.extend(collector.results[f'{var_name}{suffix}'])

        vmin, vmax = min(all_values), max(all_values)
        margin = (vmax - vmin) * 0.05

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        # ax.plot(t, collector.results[f'{var_name}_mean_q1'], 'b-', linewidth=1.5, label=r'$\Delta q_{mean1}$', alpha=1)
        ax.plot(t, collector.results[f'{var_name}_mean_q2'], '-', color='midnightblue', linewidth=1.5,
                label=r'$\Delta \bar{q}_{env}$', alpha=1)
        ax.plot(t, collector.results[f'{var_name}_q1'], '-', color='cyan', linewidth=1.5,
                label=r'$\Delta q_{RK1}$', alpha=1)
        ax.plot(t, collector.results[f'{var_name}_q2'], '-', color='green', linewidth=1.5,
                label=r'$\Delta q_{RK2}$', alpha=1)
        ax.plot(t, collector.results[f'{var_name}_q3'], '-', color='orange', linewidth=1.5,
                label=r'$\Delta q_{RK3}$', alpha=1)
        ax.plot(t, collector.results[f'{var_name}_q4'], 'r-', linewidth=1, label=r'$\Delta q_{RK4}$', alpha=1)
        # ax.plot(time_serie, norm_min_q1_list, 'g-', linewidth=1, label='Δq_min1', alpha=0.5)
        # ax.plot(time_serie, avg_mean_q3_list, 'r.-', linewidth=2, label='q_mean3', alpha=0.5)
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel(title, fontsize=16)
        # ax.set_title(f'{title} {self.start}-{self.end}', fontsize=18)
        ax.set_xticks([0, int(self.midnum * collector.n_group / collector.count), t[-1]])
        ax.set_xticklabels([self.start, self.mid_date, self.end], fontsize=16)
        ax.set_ylim(vmin - margin, vmax + margin)
        ax.legend(ncol=6, fontsize=16,
                  bbox_to_anchor=(0.5, 1.1),  # key parameter: x=0.5(centered), y=1.15(above the axes)
                  loc='center', frameon=False)
        ax.grid(True, alpha=0.3)
        fig.patch.set_alpha(1.0)  # transparent figure background
        ax.patch.set_alpha(0.0)  # transparent axes background
        plt.savefig(f'./fig/fig2/'
                    f'{title} {self.start}-{self.end} 600dpi.png', dpi=600, transparent=True,
                    bbox_inches='tight')
        # plt.show()
        plt.close(fig)

    def plot_any_curve(self, collector, variates, title, text=False):
        '''
        Args:
            collector: data collector object
            vars: list of variable names, e.g., ['var1', 'var2', 'var3']
            title: plot title and y-axis label
        '''
        t = range(collector.n_group)
        # cycle the colors if the number of curves exceeds the number of colors
        colors = self.colors[:len(variates)] if len(variates) <= len(self.colors) else self.colors * (len(variates) // len(self.colors) + 1)

        all_values = []
        for var in variates:
            all_values.extend(collector.results[f'{var}'])
        vmin, vmax = min(all_values), max(all_values)
        margin = (vmax - vmin) * 0.05

        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(111)
        
        if text and len(variates) >= 2:
            p = calculate_pearson(collector.results[variates[0]], collector.results[variates[1]])
            ax.text(0.8, 0.3, f'CC = {p:.3f}',
                    transform=ax.transAxes, fontsize=18,
                    bbox=dict(boxstyle='round', facecolor='w', alpha=0.5))  # round rounded rectangle, at (0.95, 0.3) near the lower-right corner

        for i, var in enumerate(variates):
            leg = 'S' if 'sigma' in var else '|Δq|'
            ax.plot(t, collector.results[f'{var}'], 
                    '-',
                    color=colors[i], 
                    linewidth=1.5, 
                    label=leg, 
                    alpha=1)

        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel(title, fontsize=16)

        ax.set_xticks([0, int(self.midnum * collector.n_group / collector.count), t[-1]])
        ax.set_xticklabels([self.start, self.mid_date, self.end], fontsize=16)

        ax.set_ylim(vmin - margin, vmax + margin)

        # set the legend (adjust position and number of columns by the number of curves)
        ncol = min(len(variates), 6)  # at most 6 columns
        ax.legend(ncol=ncol, fontsize=16,
                  bbox_to_anchor=(0.5, 1.1),
                  loc='center', frameon=False)

        ax.grid(True, alpha=0.3)
        fig.patch.set_alpha(1.0)
        ax.patch.set_alpha(0.0)
        plt.savefig(f'./fig/fig2/'
                    f'{title} {self.start}-{self.end} 600dpi.png', dpi=600, transparent=True,
                    bbox_inches='tight')
        plt.close(fig)

    def sigma_boxplot(self, data_lists, title, enhance_factor):
        # sigma curl
        fig, ax = plt.subplots(figsize=(6, 4))

        means = [np.mean(data) for data in data_lists]
        for label, data in zip(self.labels, data_lists):
            print(f'λ = {label}: {np.mean(data)}')

        q1_values = [np.percentile(data, 25) for data in data_lists]
        q3_values = [np.percentile(data, 75) for data in data_lists]
        ax.plot([p / enhance_factor for p in self.pos], means, 'b-', linewidth=2, label='Mean')
        ax.fill_between([p / enhance_factor for p in self.pos], q1_values, q3_values, edgecolor='none',
                        alpha=0.3, color='steelblue', label='IQR (Q1-Q3)')
        ax.boxplot(data_lists, positions=self.pos)
        # ax7.set_title(rf'Boxplot of Different $\lambda$ for '
        #               rf'$\frac{{\partial u}}{{\partial x}}-\frac{{\partial v}}{{\partial y}} + $', fontsize=16)
        ax.set_title('P (λ)')
        ax.set_xlabel(r'$\lambda$', fontsize=16)
        ax.set_ylabel('P', fontsize=16)
        # ax7.set_title('Mean and IQR across Different Groups')
        ax.set_xticks([p / enhance_factor for p in self.pos])
        ax.set_xticklabels(self.labels, fontsize=7)
        ax.legend(ncol=1, fontsize=16, loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.patch.set_alpha(1.0)
        ax.patch.set_alpha(0.0)
        plt.savefig(f'./fig/fig2/{title} 600dpi.png',
                    dpi=600, transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()


class MetricCollector:
    def __init__(self, bs_size, date_list, r_metrics, q_metrics, mean_q_metrics, motion_min, pearson):
        # q_metrics = ['q1', 'q2', 'q3', 'q4']
        # mean_q_metrics = ['mean_q1', 'mean_q2']    'min_q' min value in the ambient region
        # motion_min = ['min_id', 'RK4_motion', 'min_dq_field']
        # pearson = ['120', '60', '36']
        # r_metrics = ['r05','r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r12', 'r15']
    
        # ratio_dict: r1_ratio, r2_5_ratio, r5_ratio, r8_ratio, r10_ratio, r12_ratio, r15_ratio
        # bk_dq_dict: bk_q1, bk_q2, bk_q3, bk_q4, bk_RK4_motion, 
        # fw_dq_dict: fw_q1, fw_q2, fw_q3, fw_q4, fw_RK4_motion
        # bk_mean_dq_dict: bk_mean_q1, bk_mean_q2, bk_min_id, bk_min_dq_field
        # fw_mean_dq_dict: fw_mean_q1, fw_mean_q2, fw_min_id, fw_min_dq_field
        # sigma_dq_plot: sum_sigma, sum_dq
        # sigma_dict: last_sigma, present_sigma
        # pearson_dict: pearson120, pearson60, pearson36
        self.days = [(datetime.strptime(date, '%Y%m%d') - datetime(1993, 1, 1)).days for date in date_list]  # id
        self.date_list = list(zip(self.days, date_list))  # zip(·) iterator, iterated for once and it will be running out
        self.r_metrics = r_metrics
        self.q_metrics = q_metrics
        self.mean_q_metrics = mean_q_metrics
        self.motion_min = motion_min
        self.pearson = pearson
        self.temp = {  # q1, q2, q3, q4, q_mean1, q_mean2, min_q
            **{f'fw_{m}': [] for m in self.q_metrics + self.mean_q_metrics},
            **{f'bk_{m}': [] for m in self.q_metrics + self.mean_q_metrics},
            **{f'pearson{m}': [] for m in self.pearson},
            'last_sigma': [], 'present_sigma':[]
        }
        self.results = {
            'time_serie': [],
            **{f'{m}_list': [] for m in self.r_metrics},
            **{f'avg_bk_{m}': [] for m in self.q_metrics + self.mean_q_metrics},
            **{f'norm_bk_{m}': [] for m in self.q_metrics + self.mean_q_metrics},
            **{f'avg_fw_{m}': [] for m in self.q_metrics + self.mean_q_metrics},
            **{f'norm_fw_{m}': [] for m in self.q_metrics + self.mean_q_metrics},
            **{f'avg_fw_bk_{m}': [] for m in self.q_metrics + self.mean_q_metrics},
            **{f'norm_fw_bk_{m}': [] for m in self.q_metrics + self.mean_q_metrics},
            **{f'avg_pearson{m}': [] for m in self.pearson},
            'avg_last_sigma': [], 'avg_present_sigma':[], 'avg_last_present_sigma':[]
        }
        self.plot_results = {  # min_id, RK4_motion, min_dq_field, dq4_field
            **{f'bk_{m}': [] for m in self.motion_min},
            **{f'fw_{m}': [] for m in self.motion_min},
            **{f'bk_{m}_name': [] for m in self.motion_min},
            **{f'fw_{m}_name': [] for m in self.motion_min},
            'sum_sigma':[],'sum_dq':[], 'sum_sigma_name':[],'sum_dq_name':[], 'lamda':[], 'lamda_name':[]
        }
        self.count = 0
        self.n_group = 0
        self.bs = bs_size

    def update(self, result):  # update all the metrics
        for m in self.r_metrics:
            if f'{m}_ratio' in result:
                self.results[f'{m}_list'].extend(result[f'{m}_ratio'])
        self.count += len(result[f'{m}_ratio'])

        if 'last_sigma' in result and 'present_sigma' in result:
            self.temp['last_sigma'].append(result['last_sigma'])
            self.temp['present_sigma'].append(result['present_sigma'])
            
        for m in self.pearson:
            if f'pearson{m}' in result:
                self.temp[f'pearson{m}'].append(result[f'pearson{m}'])

        for m in self.q_metrics + self.mean_q_metrics:  # temp: q1, q2, q3, q4, mean_q1, mean_q2,
            if f'bk_{m}' in result:
                self.temp[f'bk_{m}'].append(result[f'bk_{m}'])
            if f'fw_{m}' in result:
                self.temp[f'fw_{m}'].append(result[f'fw_{m}'])

        plot_data = [(x % self.bs, day_str) for x, day_str in self.date_list
                     if x < self.count and self.count - self.bs <= x]
        for m in self.motion_min:  # 'min_id', 'RK4_motion' 'min_dq_field'
            if f'bk_{m}' in result and plot_data:
                for id, str in plot_data:
                    self.plot_results[f'bk_{m}'].append(result[f'bk_{m}'][id])
                    self.plot_results[f'bk_{m}_name'].append(str)
            if f'fw_{m}' in result and plot_data:
                for id, str in plot_data:
                    self.plot_results[f'fw_{m}'].append(result[f'fw_{m}'][id])
                    self.plot_results[f'fw_{m}_name'].append(str)
                    
        if f'sum_sigma' in result and plot_data:
            for id, str in plot_data:
                self.plot_results[f'sum_sigma'].append(result[f'sum_sigma'][id])
                self.plot_results[f'sum_sigma_name'].append(str)
        if f'sum_dq' in result and plot_data:
            for id, str in plot_data:
                self.plot_results[f'sum_dq'].append(result[f'sum_dq'][id])
                self.plot_results[f'sum_dq_name'].append(str)
        if f'lamda' in result and plot_data:
            for id, str in plot_data:
                self.plot_results[f'lamda'].append(result[f'lamda'][id])
                self.plot_results[f'lamda_name'].append(str)

    def aggregate(self):
        """Aggregate temporary data"""  # temp # last_sigma, present_sigma, pearson120, pearson60, pearson36, bk, fw: q1, q2, q3, q4, mean_q1, mean_q2,
        if self.temp.get('bk_q4') and self.temp.get('fw_q4'):
            div_bk = np.sqrt(np.mean(np.vstack(self.temp['bk_q4'])))
            div_fw = np.sqrt(np.mean(np.vstack(self.temp['fw_q4'])))
            div_fw_bk = div_bk + div_fw
        
        for key, values in self.temp.items():  # temp --> last_sigma, present_sigma, pearson120, pearson60, pearson36, bk, fw: q1, q2, q3, q4, mean_q1, mean_q2, 
            if values:
                all_values = np.vstack(values)
                if 'pearson' in key:
                    avg_val = np.mean(all_values)
                else:
                    avg_val = np.sqrt(np.mean(all_values))
                self.results[f'avg_{key}'].append(avg_val)
                
                # calc normalization
                if 'bk' in key and 'fw' not in key:
                    self.results[f'norm_{key}'].append(avg_val / div_bk)
                elif 'fw' in key and 'bk' not in key and self.temp['fw_q4']:
                    self.results[f'norm_{key}'].append(avg_val / div_fw)
                    
        for m in self.q_metrics + self.mean_q_metrics:
            if self.temp.get(f'bk_{m}') and self.temp.get(f'fw_{m}'):
                a = np.sqrt(np.mean(np.vstack(self.temp[f'bk_{m}']))) + np.sqrt(np.mean(np.vstack(self.temp[f'fw_{m}'])))
                self.results[f'avg_fw_bk_{m}'].append(a)
                self.results[f'norm_fw_bk_{m}'].append(a / div_fw_bk)
            else:
                print(f'{m} does not exist!')
                    
        if self.temp.get('last_sigma') and self.temp.get('present_sigma'):
            last_present_S = np.sqrt(np.mean(np.vstack(self.temp['last_sigma']))) + np.sqrt(np.mean(np.vstack(self.temp['present_sigma'])))
            print(last_present_S)
            self.results['avg_last_present_sigma'].append(last_present_S)
            print(f"Average of iteration {self.count + 1}: "
                  f"last + present sigma={last_present_S:.7f}")
        else:
            print('No sigma information input!!!!!!!!')
            
        if self.temp.get('pearson120') and self.temp.get('pearson36'):
            p120 = np.vstack(self.temp['pearson120'])
            p60 = np.vstack(self.temp['pearson60'])
            p36 = np.vstack(self.temp['pearson36'])
            print(f"Average of iteration {self.count + 1}: "
                  f"pearson120={np.mean(p120):.5f},"
                  f"pearson60={np.mean(p60):.5f},"
                  f"pearson36={np.mean(p36):.5f},")
        else:
            print('No pearson information input!!!!!!!!')

          
        if self.temp.get('bk_q1') and self.temp.get('bk_q4'):
            all_bk_q1 = np.vstack(self.temp['bk_q1'])
            all_bk_q2 = np.vstack(self.temp['bk_q2'])
            all_bk_q3 = np.vstack(self.temp['bk_q3'])
            all_bk_q4 = np.vstack(self.temp['bk_q4'])
            print(f"Average of iteration {self.count + 1}: "
                  f"bk q1={np.sqrt(np.mean(all_bk_q1)):.7f}, "
                  f"bk q2={np.sqrt(np.mean(all_bk_q2)):.7f}, "
                  f"bk q3={np.sqrt(np.mean(all_bk_q3)):.7f}, "
                  f"bk q4={np.sqrt(np.mean(all_bk_q4)):.7f}")
        else:
            print('No backward q information input!!!!!!!!')


        if self.temp.get('fw_q1') and self.temp.get('fw_q4'):
            all_fw_q1 = np.vstack(self.temp['fw_q1'])
            all_fw_q2 = np.vstack(self.temp['fw_q2'])
            all_fw_q3 = np.vstack(self.temp['fw_q3'])
            all_fw_q4 = np.vstack(self.temp['fw_q4'])
            print(f"Average of iteration {self.count + 1}: "
                  f"fw q1={np.sqrt(np.mean(all_fw_q1)):.7f}, "
                  f"fw q2={np.sqrt(np.mean(all_fw_q2)):.7f}, "
                  f"fw q3={np.sqrt(np.mean(all_fw_q3)):.7f}, "
                  f"fw q4={np.sqrt(np.mean(all_fw_q4)):.7f}")
        else:
            print('No forward dq information input!!!!!!!!')

        # clear temporary storage
        self.temp = {k: [] for k in self.temp}
        self.n_group += 1

    def should_aggregate(self, average_num, total_samples):
        """Decide whether aggregation is needed"""
        return (self.count % average_num == 0) or (self.count == total_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Global SLA Predictor Training Config")
    parser.add_argument('--loss_mode', type=str, default='1', help='Loss functions mode (e.g., 1, 2, 3)')
    parser.add_argument('--l2_coeff', type=float, default=1.00, help='Coefficient for loss2 terms')
    parser.add_argument('--l3_coeff', type=float, default=1.00, help='Coefficient for loss3 terms')
    parser.add_argument('--l4_coeff', type=float, default=1.00, help='Coefficient for loss4 terms')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    config = Config(args, train_shuffle=False)
    analysis_start = "1993-01-01"
    analysis_end = "2003-01-01"
    config.lead = 1
    config.batch_size = 5
    config.dataset_start = "1993-01-01"
    config.train_start = (datetime.strptime(analysis_start, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d")
    config.val_start = (datetime.strptime(analysis_end, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d")
    config.test_start = (datetime.strptime(analysis_end, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
    config.test_end = (datetime.strptime(analysis_end, "%Y-%m-%d") + timedelta(days=20)).strftime("%Y-%m-%d")
    nv = config.num_var
    sql = config.seq_len

    all_day = (datetime.strptime(analysis_end, "%Y-%m-%d") - datetime.strptime(analysis_start, "%Y-%m-%d")).days
    midnum = all_day // 2

    data_dict = load_and_process_data(config)
    device = torch.device('cuda:0')
    dm = GlobalCurrentDataModule(config, data_dict['land_mask'])
    dm.setup(stage='fit')
    train_dataset = dm.train_dataset
    bs = 5
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
    model = SwinTransformer(data_dict, config, spemb=True, img_size=(2040, 4320), patch_size=12,
                   embed_dim=192, depths=(4,), bottle_neck=(4,), num_heads=(4, 8, 4),
                   window_size=10, mlp_ratio=4., ape=False, use_checkpoint=False).to(device)

    date_list = ['19930101', '19940301', '19950401', '19960501', '19970701',
                 '19980801', '19990901', '20001001', '20011201']
    '''date_list = ['19930101', '19930102', '19930103', '19930104', '19930105',
                 '19930106', '19930107', '19930108', '19930109']  # test'''
    q_metrics = ['q1', 'q2', 'q3', 'q4']
    mean_q_metrics = ['mean_q2']  # ['mean_q1', 'mean_q2', 'min_q']
    motion_min = ['min_id', 'RK4_motion', 'min_dq_field']
    pearson = ['120', '60', '36']

    r_metrics = ['r05','r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r12', 'r15']
    positions = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15]
    labels = ['0.5','1', '2', '3', '4', '5', '6', '7', '8', '9', '12', '15']

    collector = MetricCollector(bs, date_list, r_metrics, q_metrics, mean_q_metrics, motion_min, pearson)
    VEC = ValiSigmaQ(model, config.resolution, config.expand_size, positions, r_metrics)

    P2D = Plot2D(data_dict['coord_grid'], data_dict['land_mask'], analysis_start, analysis_end, positions, labels)
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='Epoch 1/1', unit='batch'):
            seq = batch['input'].to(device)
            result = VEC(seq)
            collector.update(result)
            if collector.should_aggregate(bs * 2, all_day):
                # take the mean every 30 days
                # (must be an integer multiple of bs) interval and total samples is 3652
                collector.aggregate()
        data_lists = [collector.results[f'{r}_list'] for r in r_metrics]

    # avg_bk, norm_bk, avg_fw, norm_fw, avg_fw_bk, norm_fw_bk
    # normalization Δq
    P2D.plot_multi_curves(collector, 'norm_bk', 'Backward Δq Normalization')
    P2D.plot_multi_curves(collector, 'norm_fw', 'Forward Δq Normalization')
    P2D.plot_multi_curves(collector, 'norm_fw_bk', 'Forward + Backward Δq Normalization')
    # Mean Δq
    P2D.plot_multi_curves(collector, 'avg_bk', 'Backward Δq Mean')
    P2D.plot_multi_curves(collector, 'avg_fw', 'Forward Δq Mean')
    P2D.plot_multi_curves(collector, 'avg_fw_bk', 'Forward + Backward Δq Mean')
    # sigma boxplot
    P2D.sigma_boxplot(data_lists, 'Boxplot of Different λ for S', 1)  # Strain Deformation
    # enhace_factor enlarges as X ticks get tighter
    # sigma curve
    P2D.plot_any_curve(collector, ['avg_last_present_sigma', 'avg_fw_bk_q4'], 'S -Δq', text=True)  # Strain Deformation
    # P2D.plot_any_curve(collector, ['avg_pearson120', 'avg_pearson60', 'avg_pearson36'], 'Pearson Correlation Mean 2000')  

    # Plot cmp: motion:viridis, sigma:plasma, Δq:RdBu, id: YlGnBu, inferno, magma
    # abs q: coolwarm, dq:RdBu
    P2D.plot_subplots(collector.plot_results['sum_sigma'], collector.plot_results['sum_sigma_name'],
                     'Last + Present S', 'last + present S', 'plasma')
    P2D.plot_subplots(collector.plot_results['sum_dq'], collector.plot_results['sum_dq_name'],
                     'Last + Present |dq|', 'last + present |dq|', 'coolwarm')
   
    P2D.plot_subplots(collector.plot_results['bk_RK4_motion'], collector.plot_results['bk_RK4_motion_name'],
                     'Backward Motion (km)', 'BK RK4 motion', 'viridis')
    P2D.plot_subplots(collector.plot_results['bk_min_dq_field'], collector.plot_results['bk_min_dq_field_name'],
                     'Backward Min Δq (s⁻¹)', 'BK min Δq', 'coolwarm')
    P2D.plot_subplots(collector.plot_results['fw_min_id'], collector.plot_results['fw_min_id_name'],
                     'Forward Min Δq positions', 'FW min Δq positions', 'YlGnBu')
    P2D.plot_subplots(collector.plot_results['fw_RK4_motion'], collector.plot_results['fw_RK4_motion_name'],
                     'Forward Motion (km)', 'FW RK4 motion', 'viridis')
    P2D.plot_subplots(collector.plot_results['fw_min_dq_field'], collector.plot_results['fw_min_dq_field_name'],
                     'Forward Min Δq (s⁻¹)', 'FW min Δq', 'coolwarm')
    '''P2D.plot_subplots(collector.plot_results['lamda'], collector.plot_results['lamda_name'],
                     'λ (|S / dq|)', 'sum S div sum dq', 'seismic', draw_contour=False, norm_manual=True)'''
    
    print(f'n_group: {collector.n_group}')
    print(f'count: {collector.count}')

