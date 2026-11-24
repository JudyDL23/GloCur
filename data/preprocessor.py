# data/processor.py
import numpy as np
import torch
import torch.nn.functional as F


def global_intervalx_expand(ix, expand_size):
    """Expand the interval in the x direction."""
    ix_north = torch.flip(ix[-expand_size:], dims=[0])
    ix_new = torch.cat([ix.clone(), ix_north], dim=0)
    return ix_new


def producing_lonlat():
    """Generate longitude/latitude grid."""
    lat_v12 = torch.linspace(-80 + 1 / 12, -80 + 2040 * 1 / 12, steps=2040)
    lon_v12 = torch.linspace(-180, -180 + (4320 - 1) * 1 / 12, steps=4320)
    return lat_v12, lon_v12  # 1/12 degree


def load_and_process_data(config):
    """
    Unified interface for loading and processing data.
    Note: this function now receives a config object and no longer needs the t and t1 arguments.
    """
    resolution = config.resolution
    exp_size = config.expand_size
    region = 'global'

    domain = {'global': [-180, 180 - 1 / 12, -80 + 1 / 12, 90]}
    bttm_lonlat = domain[region]
    bottom_lon = bttm_lonlat[0]
    bottom_lat = bttm_lonlat[2]

    data_dir = config.auxiliary_dir
    date_range = config.train_range

    # Loading File Path
    sea_bed_path = f'{data_dir}Global_0.083deg_altitude.npy'
    uv_clim_path = f'{data_dir}Global_uv_clim_{date_range}.npy'
    landmask_path = f'{data_dir}land_mask.npy'
    mean_path = f'{data_dir}Global_uv_mean_{date_range}.npy'
    std_path = f'{data_dir}Global_uv_std_{date_range}.npy'

    # Loading Data
    seabed_dataset = np.load(sea_bed_path)  # original size: 2041, 4320 ---> 2040, 4320
    seabed = torch.tensor(seabed_dataset[1:, :], dtype=torch.float32)
    print(f'seabed size: {seabed.shape}')
    seabed = torch.clamp(seabed, max=0)

    uv_clim_dataset = np.load(uv_clim_path)  # original size: 2041, 4320 ---> 2040, 4320
    uv_clim = torch.tensor(uv_clim_dataset[..., 1:, :], dtype=torch.float32)
    print(f'uv_clim size: {uv_clim.shape}')

    land_mask_dataset = np.load(landmask_path)  # original size: 2041, 4320 ---> 2040, 4320
    land_mask = torch.tensor(land_mask_dataset[1:, :], dtype=torch.bool)  # cast to bool
    print(f'land_mask size: {land_mask.shape}')
    seabed[land_mask] = 0

    mean_dataset = np.load(mean_path)
    mean = torch.tensor(mean_dataset, dtype=torch.float32)
    print(f'uv mean size: {mean.shape}')

    std_dataset = np.load(std_path)
    std = torch.tensor(std_dataset, dtype=torch.float32)
    print(f'uv std size: {std.shape}')
    mean_std = torch.cat([mean, std], dim=0)
    print(f'uv mean std size: {mean_std.shape}')
    # u_mean,v_mean,u_std,v_std

    # calc lat, lon, grid
    row, col = land_mask.shape[-2:]
    lat_vector, lon_vector = producing_lonlat()

    interval_x = 6371. * torch.cos(lat_vector * torch.pi / 180) * (torch.pi / 180) * resolution  # km
    interval_y = torch.ones(1) * 6371. * (torch.pi / 180) * resolution  # km
    lon = lon_vector.unsqueeze(0).expand(row, -1).unsqueeze(0)  # 1, H, W
    lat = lat_vector.unsqueeze(1).expand(-1, col).unsqueeze(0)
    w_lat = torch.cos(lat * torch.pi / 180).squeeze(0)  # H, W
    masked_w = w_lat[~land_mask]  # (~masked)
    norm_w = masked_w / torch.sum(masked_w)  # (~mask)
    coord_grid = torch.cat([lon, lat], dim=0)

    # expand interval_x
    interval_x = global_intervalx_expand(interval_x, exp_size)

    # calc grid index
    col_ind = torch.arange(col + 2 * exp_size).repeat(row + exp_size, 1).unsqueeze(0)
    row_ind = torch.arange(row + exp_size).unsqueeze(1).expand(row + exp_size, col + 2 * exp_size).unsqueeze(0)
    ind_grid = torch.cat([col_ind, row_ind], dim=0)

    return {
        'land_mask': land_mask,
        'uv_clim': uv_clim,
        'mean_std': mean_std,
        'seabed': seabed,
        'interval_x': interval_x,
        'interval_y': interval_y,
        'coord_grid': coord_grid,
        'ind_grid': ind_grid,
        'norm_w': norm_w
    }
