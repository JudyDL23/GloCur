from validate_plot.validate_pv import *
from tqdm import tqdm
from model.LossFunc import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo
from datetime import datetime
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import torch
import numpy as np
from tqdm import tqdm
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
from matplotlib import colors
import matplotlib.ticker as mticker


# fw: forward
# bk: backward
def pltnorm(concatenated):
    vmin_tensor = concatenated.min()
    vmax_tensor = concatenated.max()  # at this point both are 0-dim tensors
    vmin = vmin_tensor.item()
    vmax = vmax_tensor.item()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    return norm


class CalcMotion(nn.Module):  # batchsize * (num variables * seq length) * latitude* longitude 6*193*505
    def __init__(self, model, resolution, exp_size, RK_list):
        super(CalcMotion, self).__init__()
        self.register_buffer('ix', model.ix, persistent=False)
        self.register_buffer('iy', model.iy, persistent=False)
        self.register_buffer('coor', model.coor, persistent=False)
        self.row, self.col = self.coor.shape[-2], self.coor.shape[-1]
        self.ix_glob = self.ix[:self.row].unsqueeze(-1).expand(-1, self.col).unsqueeze(0)
        self.register_buffer('lat', model.coor[1], persistent=False)
        self.res = resolution
        self.eps = exp_size
        self.RK1 = RK1(model)
        self.RK2 = RK2(model, self.res, self.eps, 'parallel')
        self.RK3 = RK3(model, self.res, self.eps, 'parallel')
        self.RK4 = RK4(model, self.res, self.eps, 'parallel')
        self.getIntp = GetIntp_parallel(model, resolution, exp_size)
        self.RK_list = RK_list

    def forward(self, field, vel, time, interpolated_field, date):
        plot_results = {}
        for rk in self.RK_list:
            rk_method = getattr(self, rk)  # dynamic adjustment
            grid_motion = torch.zeros(5, 2, self.row, self.col)
            for i in range(5):
                motion = rk_method(field, vel, time * i / 4, torch.zeros_like(vel))  # (km), 1 2 H W
                new_lon = motion[:, 0] / self.ix_glob * self.res + self.coor[0]
                new_lat = motion[:, 1] / self.iy * self.res + self.lat
                grid_motion[i, 0, :, :] = new_lon
                grid_motion[i, 1, :, :] = new_lat
                if i == 4:
                    y_degree = (motion[:, 1:2].clone() / self.iy) * self.res  # 1, 1, H, W
                    f_interpolated = 2 * 7.292e-5 * torch.sin((self.lat + y_degree) * torch.pi / 180)  # 1, 1, H, W
                    interpolating_field = self.getIntp(interpolated_field, motion)  # 1, 1, H, W
                    q_interpolated = interpolating_field + f_interpolated
            plot_results[f'{date}_{rk}_motion'] = grid_motion.cpu().numpy()
            plot_results[f'{date}_{rk}_q_interpolated'] = q_interpolated.squeeze()
        return plot_results


class MetricCollector:
    def __init__(self, analysis_start, bs_size, date_list, model, res, exp_size, RK_list):
        self.days = [(datetime.strptime(date, '%Y%m%d') - datetime.strptime(analysis_start, "%Y-%m-%d")).days for date in date_list]  # id
        self.date_list = list(zip(self.days, date_list))  # zip(·) iterator, iterated for once and it will be running out
        self.mask = model.land_mask
        self.ix = model.ix
        self.iy = model.iy
        self.coor = model.coor
        self.count = 0
        self.bs = bs_size
        self.CM = CalcMotion(model, res, exp_size, RK_list)
        self.dt = 86400
        self.f = 2 * 7.292e-05 * torch.sin(torch.pi * model.coor[1] / 180)
        self.bk = {}
        self.fw = {}
        self.sigma = {}
        self.uv = {}
        self.eps = exp_size

    def update(self, batch, target):  # update all the metrics
        self.count += batch.shape[0]
        plot_data = [(x % self.bs, day_str) for x, day_str in self.date_list
                     if self.count - self.bs <= x < self.count]
        if plot_data:
            for id, str in plot_data:
                seq = batch[id:id+1]
                uv_last = seq[:, 2:4, :, :]
                uv_now = seq[:, :2, :, :]
                uv_last[..., self.mask] = 0  # B, 2, H, W
                uv_now[..., self.mask] = 0
                last_field = uv_last.clone()
                now_field = uv_now.clone()
                # target
                uv_rec = torch.zeros_like(uv_last)  # B, 2, H, W
                uv_rec[..., ~self.mask] = target[:, -2:]

                # uv
                uv_all = torch.cat([seq, uv_rec],dim=1)  # B, 6, H, W

                # calc partial
                u_x, u_y, v_x, v_y = calc_grad_4order(uv_last, self.ix, self.iy, self.eps)  # batch_size * 1 * lat * lon
                curl_last = (v_x - u_y) * 0.001
                sigma_last = torch.sqrt((u_x - v_y) ** 2 + (v_x + u_y) ** 2) * 0.001  # B,1,H,W
                exp_last = torch.sqrt((u_x - v_y) ** 2) * 0.001
                shear_last = torch.sqrt((v_x + u_y) ** 2) * 0.001
                del u_x, u_y, v_x, v_y
                u_x, u_y, v_x, v_y = calc_grad_4order(uv_now, self.ix, self.iy, self.eps)  # batch_size * 1 * lat * lon
                curl_now = (v_x - u_y) * 0.001
                sigma_now = torch.sqrt((u_x - v_y) ** 2 + (v_x + u_y) ** 2) * 0.001  # B,1,H,W
                exp_now = torch.sqrt((u_x - v_y) ** 2) * 0.001
                shear_now = torch.sqrt((v_x + u_y) ** 2) * 0.001
                del u_x, u_y, v_x, v_y

                bk_results = self.CM(last_field, uv_last, -self.dt, curl_last, str)
                bk_results[f'{str}_q_true'] = (curl_now + self.f).squeeze()
                fw_results = self.CM(now_field, uv_now, self.dt, curl_now, str)
                fw_results[f'{str}_q_true'] = (curl_last + self.f).squeeze()
                self.sigma[f'{str}_sigma_sum'] = (sigma_last + sigma_now).squeeze()
                self.sigma[f'{str}_sigma_last'] = sigma_last.squeeze()
                self.sigma[f'{str}_sigma_now'] = sigma_now.squeeze()
                self.sigma[f'{str}_shear_last'] = shear_last.squeeze()
                self.sigma[f'{str}_shear_now'] = shear_now.squeeze()
                self.sigma[f'{str}_exp_last'] = exp_last.squeeze()
                self.sigma[f'{str}_exp_now'] = exp_now.squeeze()
                self.uv[f'{str}_uv'] = uv_all.squeeze()
                self.bk = {**self.bk, **bk_results}
                self.fw = {**self.fw, **fw_results}


class PlotMedTrack:
    def __init__(self, coor, mask, interval_x, interval_y, plot_step):
        self.coor = coor  #
        self.land = mask
        self.ix = interval_x.unsqueeze(-1).expand(-1, mask.shape[-1]).unsqueeze(0)
        self.iy = interval_y
        self.dt = 86400
        row = 2040  # 193
        col = 4320  # 505
        bottom_lat = -80 + 1 / 12
        bottom_lon = -180
        start_lat = -80 + 1 / 12  # 35
        start_lon = -180  # 0
        self.sample_points = []
        for i in range(0, row, plot_step):
            for j in range(0, col, plot_step):
                if not self.land[i, j]:
                    self.sample_points.append((i, j))
        self.lon_rank = slice(int(12 * (start_lon - bottom_lon)), int(12 * (start_lon - bottom_lon) + col))
        self.lat_rank = slice(int(12 * (start_lat - bottom_lat)), int(12 * (start_lat - bottom_lat) + row))
        self.x = self.coor[0, self.lat_rank, self.lon_rank].cpu().numpy()  # longitude
        self.y = self.coor[1, self.lat_rank, self.lon_rank].cpu().numpy()  # latitude

    def plot_map(self, data, norm, map_color, title, style='globe'):
        # set the projection
        if style == 'globe':
            projection = ccrs.Orthographic(central_longitude=160, central_latitude=20)  # 160, 20  -40, 10
            data_transform = ccrs.PlateCarree()
            fig_size = (6, 6)  # detail plot 12, 12 ; conceptual plot (6, 6)
            add_gridlines = False
        elif style == 'mollweide':
            # elliptical projection (Mollweide), global extent, longitude -180 to 180
            projection = ccrs.Mollweide(central_longitude=180)
            data_transform = ccrs.PlateCarree()
            fig_size = (8, 5)  # ratio suited to the elliptical shape
            add_gridlines = False
        elif style == 'flat':
            projection = ccrs.PlateCarree(central_longitude=180)
            data_transform = ccrs.PlateCarree()
            fig_size = (18, 14)  # enlarged flat map (change back to (16, 8) if you prefer the original ratio)
            add_gridlines = True

        fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': projection})
        im = ax.pcolormesh(self.x, self.y, data,
                           cmap=map_color,
                           transform=data_transform,
                           shading='auto', norm=norm, zorder=1)

        # in globe mode, force the global extent to avoid showing only a local region
        if style in ['globe', 'mollweide']:
            ax.set_global()

        # lon_min, lon_max, lat_min, lat_max = lon.min(), lon.max(), lat.min(), lat.max()
        # for a flat map that needs a restricted extent, uncomment the following line
        # if style == 'flat':
        #     ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        ax.coastlines(resolution='50m', linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=2)
        ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=0)

        # cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        # cbar.set_label(f'{title} (m/s)', labelpad=15, y=1.05, rotation=0)

        # set background transparent
        fig.patch.set_alpha(0.0)  # transparent figure background
        ax.patch.set_alpha(0.0)  # transparent axes background

        # ax.set_title(title, fontsize=12)
        # fig.suptitle(r'{}',
        #              y=0.1,
        #              fontsize=18)

        if style == 'flat':
            ax.set_aspect(1.2)
        else:
            ax.set_aspect('auto')  # let cartopy handle the ratio automatically in globe mode

        # decide whether to add gridlines based on the configuration
        if add_gridlines:
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0., linestyle='-',
                              xlocs=mticker.MultipleLocator(5),
                              ylocs=mticker.MultipleLocator(5))
            gl.top_labels = False
            gl.right_labels = False

        plt.savefig(f'./fig'
                    f'/{title}_pacific.png', dpi=600, transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        # plt.show()
        plt.close(fig)

    def plot_track(self, grid_motion, color, title, center_lon=None, center_lat=None,
               zoom_width=None, zoom_height=None, style='globe'):
        # set the projection
        if style == 'globe':
            projection = ccrs.Orthographic(central_longitude=160, central_latitude=20)  #  160, 20  -40, 10
            data_transform = ccrs.PlateCarree()
            fig_size = (6, 6)
            add_gridlines = False
        else:
            projection = ccrs.PlateCarree()
            data_transform = ccrs.PlateCarree()
            fig_size = (18, 14)
            add_gridlines = True

        fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': projection})

        if style == 'globe':
            ax.set_global()

        # plot the trajectories
        for i, j in self.sample_points:
            track_x = grid_motion[:, 0, i, j]
            track_y = grid_motion[:, 1, i, j]

            ax.plot(track_x, track_y, color=color, linewidth=0.5, alpha=1,
                    transform=data_transform, zorder=4)
            ax.scatter(track_x[0], track_y[0], color='black', s=0.5,
                       transform=data_transform, zorder=5)
            ax.scatter(track_x[-1], track_y[-1], color=color, s=0.5,
                       transform=data_transform, zorder=5)

        # draw the rectangle box
        if center_lon is not None:
            lon_min = center_lon - zoom_width / 2
            lon_max = center_lon + zoom_width / 2
            lat_min = center_lat - zoom_height / 2
            lat_max = center_lat + zoom_height / 2

            if style == 'globe':
                # spherical mode: draw a great-circle rectangle box
                geod = cgeo.Geodesic()
                corners = [(lon_min, lat_min), (lon_max, lat_min),
                           (lon_max, lat_max), (lon_min, lat_max)]
                for i in range(4):
                    start = corners[i]
                    end = corners[(i + 1) % 4]
                    # compute distance and initial azimuth
                    res = geod.inverse(start, end)          # returns [[dist, az1, az2]]
                    dist = res[0, 0]
                    az = res[0, 1]
                    # generate path points
                    n_pts = 50
                    dists = np.linspace(0, dist, n_pts)
                    pts = geod.direct(start, az, dists)     # returns (n_pts, 3): [lon, lat, az2]
                    ax.plot(pts[:, 0], pts[:, 1], color='red', linewidth=1, linestyle='-',
                            transform=ccrs.Geodetic(), zorder=6)
            else:
                # flat mode: straight-line rectangle box
                box_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
                box_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
                ax.plot(box_lons, box_lats, color='red', linewidth=1, linestyle='-',
                        transform=ccrs.PlateCarree(), zorder=6)

        # map decoration
        ax.add_feature(cfeature.OCEAN, color='white', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#D3D3D3', edgecolor='black', linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=3)

        if style == 'flat':
            ax.set_aspect(1.2)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0)
        else:
            ax.set_aspect('auto')

        plt.savefig(f'./fig/{title}_track.png',
                    dpi=600, transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_track_geo_zoom(self, grid_motion, color, title, center_lon=None, center_lat=None,
                            zoom_width=5.0, zoom_height=5.0):

        lon_min = center_lon - zoom_width / 2
        lon_max = center_lon + zoom_width / 2
        lat_min = center_lat - zoom_height / 2
        lat_max = center_lat + zoom_height / 2

        # find the sample points within this geographic extent
        zoom_points = []
        for i, j in self.sample_points:
            lon = self.x[i, j]
            lat = self.y[i, j]
            if (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max):
                zoom_points.append((i, j))

        if not zoom_points:
            print(f"Warning: No sample points in zoom region ({lon_min:.2f}-{lon_max:.2f}°E, "
                  f"{lat_min:.2f}-{lat_max:.2f}°N)")
            return

        # create the figure
        fig, ax = plt.subplots(figsize=(4, 4),
                               subplot_kw={'projection': ccrs.PlateCarree()})

        # plot the trajectories
        for i, j in zoom_points:
            track_x = grid_motion[:, 0, i, j]
            track_y = grid_motion[:, 1, i, j]

            ax.plot(track_x, track_y,
                    color=color,
                    linewidth=1,
                    alpha=1,
                    transform=ccrs.PlateCarree(),
                    zorder=1)

            ax.scatter(track_x[0], track_y[0],
                       color='black', s=12,
                       transform=ccrs.PlateCarree(),
                       zorder=2)

            ax.scatter(track_x[-1], track_y[-1],
                       color=color, s=12,
                       transform=ccrs.PlateCarree(),
                       zorder=2)

        # add map elements
        ax.add_feature(cfeature.OCEAN, color='white', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#D3D3D3', edgecolor='black', linewidth=0.5, zorder=4)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=5)

        # set the geographic extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        lon_ticks = np.arange(np.floor(lon_min), np.ceil(lon_max) + 1, 2)
        lat_ticks = np.arange(np.floor(lat_min), np.ceil(lat_max) + 1, 2)

        # add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                          xlocs=lon_ticks, ylocs=lat_ticks)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 16}  # set longitude label font size to 16
        gl.ylabel_style = {'size': 16}  # set latitude label font size to 16

        # set the title
        # ax.set_title(f'{lon_min:.2f}-{lon_max:.2f}°E, {lat_min:.2f}-{lat_max:.2f}°N',
        #              fontsize=12)
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(1)  # add an outer border around the whole figure
        plt.savefig(f'./fig/{title}_zoom_track.png',
                    dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()


parser = argparse.ArgumentParser(description="Global SLA Predictor Training Config")
parser.add_argument('--loss_mode', type=str, default='1', help='Loss functions mode (e.g., 1, 2, 3)')
parser.add_argument('--l2_coeff', type=float, default=1.00, help='Coefficient for loss2 terms')
parser.add_argument('--l3_coeff', type=float, default=1.00, help='Coefficient for loss3 terms')
parser.add_argument('--l4_coeff', type=float, default=1.00, help='Coefficient for loss4 terms')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()
config = Config(args, train_shuffle=False)

analysis_start = "1994-01-01"
analysis_end = "1994-02-01"

config.lead = 1
config.batch_size = 1
config.dataset_start = "1993-01-01"
config.train_start = (datetime.strptime(analysis_start, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")  # 2nd day of seq, + 1 day, not 2 days!
config.val_start = (datetime.strptime(analysis_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
config.test_start = (datetime.strptime(analysis_end, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
config.test_end = (datetime.strptime(analysis_end, "%Y-%m-%d") + timedelta(days=20)).strftime("%Y-%m-%d")
nv = config.num_var
sql = config.seq_len

data_dict = load_and_process_data(config)
device = torch.device('cuda:0')
dm = GlobalCurrentDataModule(config, data_dict['land_mask'])
dm.setup(stage='fit')
train_dataset = dm.train_dataset
bs = 1
land_cpu = data_dict['land_mask']
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
model = SwinTransformer(data_dict, config, spemb=True, img_size=(2040, 4320), patch_size=12,
                        embed_dim=192, depths=(4,), bottle_neck=(4,), num_heads=(4, 8, 4),
                        window_size=10, mlp_ratio=4., ape=False, use_checkpoint=False).to(device)


with torch.no_grad():
    date_list = ['19940115']  # 20190115
    RK_list = ['RK4']
    MC = MetricCollector(analysis_start, bs, date_list, model, config.resolution, config.expand_size, RK_list)
    for batch in tqdm(train_loader, desc='Epoch 1/1', unit='batch'):
        seq = batch['input'].to(device)
        target = batch['target'].to(device)
        MC.update(seq, target)
    # Plot
    PMT = PlotMedTrack(model.coor, model.land_mask, model.ix, model.iy, plot_step=12)
    for date in date_list:
        for rk in RK_list:
            print(date)
            print(rk)
            cen_lat = 35
            cen_lon = 145
            # -77 32, 6, 6 Atlantic
            # 145, 35, 6, 6 Pacific
           
            '''PMT.plot_track(MC.bk[f'{date}_{rk}_motion'], '#1E90FF', f'{date}_{rk}_backward_track',
                           center_lon=cen_lon, center_lat=cen_lat,
                           zoom_width=6, zoom_height=6
                           )
            PMT.plot_track(MC.fw[f'{date}_{rk}_motion'], '#FF1493', f'{date}_{rk}_forward_track',
                           center_lon=cen_lon, center_lat=cen_lat,
                           zoom_width=6, zoom_height=6
                           )  # DeepPink  #FF1493 '''
            
            PMT.plot_track_geo_zoom(MC.bk[f'{date}_{rk}_motion'], '#1E90FF', f'{date}_{rk}_backward_track_zoom',
                                    center_lon=cen_lon, center_lat=cen_lat,
                                    zoom_width=6, zoom_height=6
                                    )  # DoderBlue  #1E90FF
            PMT.plot_track_geo_zoom(MC.fw[f'{date}_{rk}_motion'], '#FF1493', f'{date}_{rk}_forward_track_zoom',
                                    center_lon=cen_lon, center_lat=cen_lat,
                                    zoom_width=6, zoom_height=6
                                    )
                                    
           
            '''# dq
            bk_q_intpl = MC.bk[f'{date}_{rk}_q_interpolated'].cpu()
            bk_q_true = MC.bk[f'{date}_q_true'].cpu()
            bk_q_true[land_cpu] = 0
            fw_q_intpl = MC.fw[f'{date}_{rk}_q_interpolated'].cpu()
            fw_q_intpl[land_cpu] = 0
            fw_q_true = MC.fw[f'{date}_q_true'].cpu()
            fw_q_true[land_cpu] = 0
            # abs dq
            abs_dq_now = (bk_q_true - bk_q_intpl).abs()
            abs_dq_last = (fw_q_true - fw_q_intpl).abs()
            abs_dq_sum = abs_dq_now + abs_dq_last
            
            # sigma
            sigma_sum = MC.sigma[f'{date}_sigma_sum'].cpu()
            sigma_sum[land_cpu] = 0
            sigma_last = MC.sigma[f'{date}_sigma_last'].cpu()
            sigma_last[land_cpu] = 0
            sigma_now = MC.sigma[f'{date}_sigma_now'].cpu()
            sigma_now[land_cpu] = 0
            
            shear_last = MC.sigma[f'{date}_shear_last'].cpu()
            shear_last[land_cpu] = 0
            shear_now = MC.sigma[f'{date}_shear_now'].cpu()
            shear_now[land_cpu] = 0
            
            exp_last = MC.sigma[f'{date}_exp_last'].cpu()
            exp_last[land_cpu] = 0
            exp_now = MC.sigma[f'{date}_exp_now'].cpu()
            exp_now[land_cpu] = 0
            
            strain_norm = create_enhanced_norm(torch.cat([shear_now[~land_cpu],shear_last[~land_cpu], exp_now[~land_cpu], 
                                                          exp_last[~land_cpu]]),symmetric=False, enhance_factor=1)
            PMT.plot_map(shear_now.numpy(), strain_norm, 'plasma', f'{date}_shear_now')
            PMT.plot_map(shear_last.numpy(), strain_norm, 'plasma', f'{date}_shear_last')
            PMT.plot_map(exp_now.numpy(), strain_norm, 'plasma', f'{date}_exp_now')
            PMT.plot_map(exp_last.numpy(), strain_norm, 'plasma', f'{date}_exp_last')
           
            # uv
            uv_all = MC.uv[f'{date}_uv'].cpu()  # B, H, W
            uv_all[..., land_cpu] = 0
            # norm
            sigma_norm = create_enhanced_norm(torch.cat([sigma_now[~land_cpu],
                                                         sigma_last[~land_cpu],
                                                         sigma_sum[~land_cpu]]),
                                              symmetric=False, enhance_factor=1)
            q_norm = create_enhanced_norm(torch.cat([bk_q_intpl[~land_cpu],
                                                     bk_q_true[~land_cpu],
                                                     fw_q_intpl[~land_cpu],
                                                     fw_q_true[~land_cpu]]),
                                          symmetric=True, enhance_factor=1.3)
            absdq_norm = create_enhanced_norm(torch.cat([abs_dq_now[~land_cpu],
                                                         abs_dq_last[~land_cpu],
                                                         abs_dq_sum[~land_cpu]]),
                                              symmetric=False, enhance_factor=1)  # 1
            
            uv_norm = create_enhanced_norm(uv_all[..., ~land_cpu].flatten(),
                                          symmetric=True, enhance_factor=1)  # 1

            PMT.plot_map(bk_q_intpl.numpy(), q_norm, 'RdBu_r', f'{date}_backward_q_interpolated')
            PMT.plot_map(bk_q_true.numpy(), q_norm, 'RdBu_r', f'{date}_backward_q_true')
            PMT.plot_map(fw_q_intpl.numpy(), q_norm, 'RdBu_r', f'{date}_forward_q_interpolated')
            PMT.plot_map(fw_q_true.numpy(), q_norm, 'RdBu_r', f'{date}_forward_q_true')
            # sigma
            PMT.plot_map(sigma_now.numpy(), sigma_norm, 'plasma', f'{date}_sigma_now')
            PMT.plot_map(sigma_last.numpy(), sigma_norm, 'plasma', f'{date}_sigma_last')
            PMT.plot_map(sigma_sum.numpy(), sigma_norm, 'plasma', f'{date}_sigma_sum')
            
            PMT.plot_map(abs_dq_now.numpy(), absdq_norm, 'RdBu_r', f'{date}_abs_dq_now')  # coolwarm
            PMT.plot_map(abs_dq_last.numpy(), absdq_norm, 'RdBu_r', f'{date}_abs_dq_last')  # coolwarm
            PMT.plot_map(abs_dq_sum.numpy(), absdq_norm, 'RdBu_r', f'{date}_abs_dq_sum')  # coolwarm
           
            # uv
            PMT.plot_map(uv_all[2].numpy(), uv_norm, 'Spectral_r', f'{date}_u_last', style='mollweide')
            PMT.plot_map(uv_all[0].numpy(), uv_norm, 'Spectral_r', f'{date}_u_now', style='mollweide')
            PMT.plot_map(uv_all[4].numpy(), uv_norm, 'Spectral_r', f'{date}_u_target', style='mollweide')

            PMT.plot_map(uv_all[3].numpy(), uv_norm, 'Spectral_r', f'{date}_v_last', style='mollweide')
            PMT.plot_map(uv_all[1].numpy(), uv_norm, 'Spectral_r', f'{date}_v_now', style='mollweide')
            PMT.plot_map(uv_all[5].numpy(), uv_norm, 'Spectral_r', f'{date}_v_target', style='mollweide')'''
