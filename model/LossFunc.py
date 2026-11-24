# losses/physics_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def global_expand(x, expand_size):
    """Global expansion function (used for boundary handling)."""
    width = x.shape[-1]
    north_slice = torch.flip(x.clone()[..., -expand_size:, :], dims=[-2])
    north_shift = torch.roll(north_slice, shifts=width // 2, dims=-1)
    world_expansion = torch.cat([x.clone(), north_shift], dim=-2)
    # x = F.pad(world_expansion, (expand_size, expand_size, 0, 0), mode='circular')  # (left, right, up, down)
    x_padded = torch.cat([
      world_expansion[..., -expand_size:],
      world_expansion,
      world_expansion[..., :expand_size]
    ], dim=-1)
    return x_padded
  
  
def global_expand_longitude(x, expand_size):
    """Global expansion function (used for boundary handling)."""
    # x = F.pad(world_expansion, (expand_size, expand_size, 0, 0), mode='circular')  # (left, right, up, down)
    x_padded = torch.cat([
        x[..., -expand_size:],
        x,
        x[..., :expand_size]
    ], dim=-1)
    return x_padded
  
  
class LocalPearsonLoss(nn.Module):
    def __init__(self, land_mask, kernel_size, stride, min_pixels):
        """
        Local-window masked Pearson correlation loss (pure convolution accelerated version).
        :param land_mask: [H, W] land=1, ocean=0
        :param kernel_size: local window size (36 recommended at 1/12 degree, about 324 km)
        :param stride: sliding step kernel_size // 2
        :param min_pixels: minimum number of valid ocean points in a window; windows below this are not computed (to avoid spurious correlations)
        """
        super().__init__()
        self.kernel = kernel_size
        self.stride = stride
        self.min_pixels = min_pixels
        # ocean mask
        ocean = (1.0 - land_mask.float()).unsqueeze(0).unsqueeze(0)
        self.register_buffer('ocean_mask', global_expand_longitude(ocean, stride))

        conv_weight = torch.ones(1, 1, kernel_size, kernel_size).to(self.ocean_mask.device)
        self.register_buffer('conv_weight', conv_weight)
        self.register_buffer('window_N', F.conv2d(self.ocean_mask, self.conv_weight, stride=stride))
        # 1, 1, H_out, W_out

        # lat-weights
        H_orig = land_mask.shape[0]
        lats = torch.linspace(-80 + 1/12, 90, H_orig).to(self.ocean_mask.device)
        cos_lats = torch.cos(torch.deg2rad(lats))  # [H_orig]
        # center latitudes
        H_out = self.window_N.shape[2]  # 0, 1, 2....
        out_center_indices = torch.arange(H_out) * stride + (kernel_size // 2)
        # out_center_indices = out_center_indices.clamp(max=H_orig - 1)
        out_cos_lats = cos_lats[out_center_indices]  # [H_out]
        lat_weight_map = out_cos_lats.view(1, 1, H_out, 1)
        self.register_buffer('lat_weight', lat_weight_map)

    def forward(self, x, y):
        """
        :param x: [B, 1, H, W] predicted delta_q
        :param y: [B, 1, H, W] predicted S
        """
        mask = self.ocean_mask

        # === Step 1: mask land and compute local sums ===
        x_m = global_expand_longitude(x, self.stride) * mask * 1e5
        y_m = global_expand_longitude(y, self.stride) * mask * 1e5

        sum_x = F.conv2d(x_m, self.conv_weight, stride=self.stride)
        sum_y = F.conv2d(y_m, self.conv_weight, stride=self.stride)
        sum_xx = F.conv2d(x_m * x_m, self.conv_weight, stride=self.stride)
        sum_yy = F.conv2d(y_m * y_m, self.conv_weight, stride=self.stride)
        sum_xy = F.conv2d(x_m * y_m, self.conv_weight, stride=self.stride)
        N = self.window_N  # 1, 1, H_out, W_out

        # === Step 2: compute local Pearson via the algebraic method (all on low-resolution feature maps, no interpolation) ===
        # Var(X) = E[X^2] - (E[X])^2 = (sum_xx / N) - (sum_x / N)^2
        # Simplified to: (N * sum_xx - sum_x^2) / N^2
        # Cov(X,Y) follows the same pattern.

        # Clamp to avoid a negative denominator caused by floating-point precision
        var_x = (N * sum_xx - sum_x * sum_x).clamp(min=1e-5)
        var_y = (N * sum_yy - sum_y * sum_y).clamp(min=1e-5)
        cov_xy = (N * sum_xy - sum_x * sum_y)

        # r = cov_xy / sqrt(var_x * var_y)
        # Note: the N^2 in the denominator cancels out perfectly here!
        local_corr = cov_xy / torch.sqrt(var_x * var_y)  # [B, 1, H', W']

        # === Step 3: build the final composite weights ===
        # Condition 1: there must be enough ocean points (filter out coastal spurious correlations)
        # Condition 2: there must be enough physical variance (correlation is meaningless if S is flat)
        valid_mask = (N >= self.min_pixels) & (var_x > 1e-5) & (var_y > 1e-5)

        # Final weight = latitude cosine weight * valid-window mask
        final_weights = self.lat_weight * valid_mask.float()
        # 1, 1, H_out, W_out

        # === Step 4: latitude-weighted global average ===
        total_weight = final_weights.sum() + 1e-5
        global_weighted_corr = torch.sum(local_corr * final_weights, dim=(-2, -1)) / total_weight
        # [B, 1, H', W'] * [1, 1, H', W'] ---> [B, 1, H', W']---> [B, 1]

        # Loss: maximize positive correlation
        loss = 1.0 - global_weighted_corr
        return loss
  

def deterministic_grid_sample_2d(feat, grid, padding_mode='zeros'):
    eps = 1e-6
    """
    Deterministic 2D grid sampling, fixing dimension mismatch and NaN backpropagation issues.
    Args:
        feat: input feature tensor, shape [B, C, H, W] or [B, H, W]
        grid: sampling grid, shape [B, H_out, W_out, 2] (xy coordinates in range [-1, 1])
        padding_mode: boundary padding mode, 'zeros' or 'border'

    Returns:
        sampled features, shape [B, C, H_out, W_out] or [B, H_out, W_out]
    """
    # 1. fileter nan & inf
    if torch.isnan(feat).any() or torch.isinf(feat).any():
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
    if torch.isnan(grid).any() or torch.isinf(grid).any():
        grid = torch.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=-1.0)

    if feat.dim() == 3:
        feat = feat.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
    B, C, H, W = feat.shape
    Bg, Hg, Wg, _ = grid.shape

    # Normalized coordinates -> pixel coordinates
    grid = torch.clamp(grid, -1.0 + eps, 1.0 - eps)
    x = (grid[..., 0] + 1) * (W - 1) / 2
    y = (grid[..., 1] + 1) * (H - 1) / 2

    x = torch.clamp(x, 0.0, float(W - 1))
    y = torch.clamp(y, 0.0, float(H - 1))

    # Four neighboring integer points
    x0 = torch.floor(x).long()
    x1 = torch.clamp(x0 + 1, 0, W - 1)
    y0 = torch.floor(y).long()
    y1 = torch.clamp(y0 + 1, 0, H - 1)

    # Weights (no unsqueeze, keep [B, H, W], add the channel dimension at the end)
    wx = x - x0.float()
    wy = y - y0.float()
    w00 = (1 - wy) * (1 - wx)
    w01 = (1 - wy) * wx
    w10 = wy * (1 - wx)
    w11 = wy * wx

    # Minimal-memory indexing: expand the channel dimension only when gathering
    def sample(y_idx, x_idx):
        # y_idx, x_idx: [B, H, W]
        idx_y = y_idx.view(B, 1, Hg, Wg).expand(B, C, Hg, Wg)
        idx_x = x_idx.view(B, 1, Hg, Wg).expand(B, C, Hg, Wg)
        fy = torch.gather(feat, 2, idx_y)
        fxy = torch.gather(fy, 3, idx_x)
        return fxy

    # Memory-critical: sample, accumulate, and free one at a time
    out = sample(y0, x0) * w00.unsqueeze(1)
    out += sample(y0, x1) * w01.unsqueeze(1)
    out += sample(y1, x0) * w10.unsqueeze(1)
    out += sample(y1, x1) * w11.unsqueeze(1)

    # interpolated
    out = out.clamp(-1e6, 1e6)
    out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    return out


class GetIntp_parallel(nn.Module):
    """Interpolation module"""
    def __init__(self, model, resolution, expand_size):
        super(GetIntp_parallel, self).__init__()
        self.res = resolution
        self.exp_size = expand_size
        self.register_buffer('mask', model.land_mask, persistent=False)
        self.register_buffer('coor', global_expand(model.coor, expand_size), persistent=False)
        self.register_buffer('ind', model.ind, persistent=False)
        self.register_buffer('iy', model.iy, persistent=False)

    def forward(self, field, motion):
        motion_width_expand = F.pad(motion.clone(), (self.exp_size, self.exp_size, 0, 0))  # default 0
        # left, right, up, down
        motion_expand = F.pad(motion_width_expand, (0, 0, 0, self.exp_size))
        field_expand = global_expand(field, self.exp_size)

        H, W = field_expand.shape[-2:]  # row, col
        y_motion = motion_expand[:, 1:2, :, :] / self.iy  # batch_size * 1 * lat * col
        y_input = self.ind[1:2, :, :] + y_motion
        y_grid = 2.0 * y_input / (H - 1) - 1.0
        new_lat = self.coor[1, :, :] + y_motion * self.res  # lon, lat
        i_x = 6371. * torch.cos(new_lat * torch.pi / 180) * torch.pi / 180 * self.res

        x_motion = motion_expand[:, :1, ...] / i_x
        x_input = self.ind[:1, :, :] + x_motion
        x_grid = 2.0 * x_input / (W - 1) - 1.0
        xy_grid = torch.cat([x_grid, y_grid], dim=1).permute(0, 2, 3, 1)
        new_field = F.grid_sample(field_expand, xy_grid, padding_mode="zeros", mode="bilinear", align_corners=True)
        return new_field[..., :-self.exp_size, self.exp_size:-self.exp_size]


class GetIntp_deterministic(nn.Module):
    """Interpolation module"""
    def __init__(self, model, resolution, expand_size):
        super(GetIntp_deterministic, self).__init__()
        self.res = resolution
        self.exp_size = expand_size
        self.register_buffer('mask', model.land_mask, persistent=False)
        self.register_buffer('coor', global_expand(model.coor, expand_size), persistent=False)
        self.register_buffer('ind', model.ind, persistent=False)
        self.register_buffer('iy', model.iy, persistent=False)

    def forward(self, field, motion):
        motion_width_expand = F.pad(motion.clone(), (self.exp_size, self.exp_size, 0, 0))  # default 0
        motion_expand = F.pad(motion_width_expand, (0, 0, 0, self.exp_size))
        field_expand = global_expand(field, self.exp_size)

        H, W = field_expand.shape[-2:]  # row, col
        y_motion = motion_expand[:, 1:2, :, :] / self.iy  # batch_size * 1 * lat * col
        y_input = self.ind[1:2, :, :] + y_motion
        y_grid = 2.0 * y_input / (H - 1) - 1.0
        new_lat = self.coor[1, :, :] + y_motion * self.res
        i_x = 6371. * torch.cos(new_lat * torch.pi / 180) * torch.pi / 180 * self.res

        x_motion = motion_expand[:, :1, ...] / i_x
        x_input = self.ind[:1, :, :] + x_motion
        x_grid = 2.0 * x_input / (W - 1) - 1.0
        xy_grid = torch.cat([x_grid, y_grid], dim=1).permute(0, 2, 3, 1)
        new_field = deterministic_grid_sample_2d(field_expand, xy_grid)
        return new_field[..., :-self.exp_size, self.exp_size:-self.exp_size]


def calc_grad_4order(uv, interval_x, interval_y, expand_size):
    """Fourth-order central difference gradient computation"""
    uv = global_expand(uv, expand_size)
    u = uv[:, :1, :, :]
    v = uv[:, 1:2, :, :]

    i_x = interval_x.view(1, 1, -1, 1)
    kernel_4th_order = torch.tensor([1, -8, 0, 8, -1], dtype=torch.float32, device=uv.device) / 12.0

    # x grad
    kernel_x = kernel_4th_order.view(1, 1, 1, 5)
    u_grad_x = F.conv2d(u, kernel_x, padding=(0, 2))  # vertical, horizontal
    v_grad_x = F.conv2d(v, kernel_x, padding=(0, 2))  # (H, W)
    u_x = u_grad_x / i_x
    v_x = v_grad_x / i_x

    # y grad
    kernel_y = kernel_4th_order.view(1, 1, 5, 1)
    u_grad_y = F.conv2d(u, kernel_y, padding=(2, 0))
    v_grad_y = F.conv2d(v, kernel_y, padding=(2, 0))
    u_y = u_grad_y / interval_y
    v_y = v_grad_y / interval_y

    out_slice = (slice(None), slice(None), slice(None, -expand_size), slice(expand_size, -expand_size))
    res = u_x[out_slice], u_y[out_slice], v_x[out_slice], v_y[out_slice]
    del uv, u, v, u_x, u_y, v_x, v_y
    return res


class Trace(nn.Module):
    """Tracking Module"""
    def __init__(self):
        super(Trace, self).__init__()
        
    def forward(self, uv, dt, *_):
        return uv * dt / 1000.  # km


class RK1(nn.Module):
    """1th order Runge-Kutta Integrator"""
    def __init__(self, model):
        super(RK1, self).__init__()
        self.trace = Trace()

    def forward(self, _, vel, dt, previous_motion):
        motion = self.trace(vel, dt)
        return previous_motion + motion


class RK2(nn.Module):
    """2th order Runge-Kutta Integrator"""
    def __init__(self, model, resolution, exp_size, mode):
        super(RK2, self).__init__()
        if mode == 'parallel':
            self.getIntp = GetIntp_parallel(model, resolution, exp_size)
        elif mode == 'deterministic':
            self.getIntp = GetIntp_deterministic(model, resolution, exp_size)
        self.trace = Trace()

    def forward(self, samp, vel, dt, previous_motion):
        motion = self.trace(vel, dt)
        intpl1 = self.getIntp(samp, previous_motion + motion)
        vel1 = intpl1[:, :2, ...]

        motion = dt * (vel + vel1)/(2. * 1000)
        return previous_motion + motion


class RK3(nn.Module):
    """3th order Runge-Kutta Integrator"""
    def __init__(self, model, resolution, exp_size, mode):
        super(RK3, self).__init__()
        if mode == 'parallel':
            self.getIntp = GetIntp_parallel(model, resolution, exp_size)
        elif mode == 'deterministic':
            self.getIntp = GetIntp_deterministic(model, resolution, exp_size)
        self.trace = Trace()

    def forward(self, samp, vel, dt, previous_motion):
        motion = self.trace(vel, 0.5 * dt)
        intpl1 = self.getIntp(samp, previous_motion + motion)
        vel1 = intpl1[:, :2, ...]

        motion = self.trace(vel1, 0.75 * dt)
        intpl2 = self.getIntp(samp, previous_motion + motion)
        vel2 = intpl2[:, :2, ...]

        motion = dt * (2 * vel + 3 * vel1 + 4 * vel2) / (9. * 1000)
        return previous_motion + motion


class RK4(nn.Module):
    """4th order Runge-Kutta Integrator"""
    def __init__(self, model, resolution, exp_size, mode):
        super(RK4, self).__init__()
        if mode == 'parallel':
            self.getIntp = GetIntp_parallel(model, resolution, exp_size)
        elif mode == 'deterministic':
            self.getIntp = GetIntp_deterministic(model, resolution, exp_size)
        self.trace = Trace()

    def forward(self, samp, vel, dt, previous_motion):
        motion = self.trace(vel, 0.5 * dt)
        intpl = self.getIntp(samp, previous_motion + motion)
        vel1 = intpl[:, :2, ...]
        del motion, intpl

        motion = self.trace(vel1, 0.5 * dt)
        intpl = self.getIntp(samp, previous_motion + motion)
        vel2 = intpl[:, :2, ...]
        del motion, intpl

        motion = self.trace(vel2, dt)
        intpl = self.getIntp(samp, previous_motion + motion)
        vel3 = intpl[:, :2, ...]
        del motion, intpl
        
        motion = dt * (vel + 2 * vel1 + 2 * vel2 + vel3) / (6. * 1000)
        return previous_motion + motion


class DeltaPV(nn.Module):
    def __init__(self, model, resolution, exp_size, mode):
        super(DeltaPV, self).__init__()
        if mode == 'parallel':
            self.getIntp = GetIntp_parallel(model, resolution, exp_size)
        elif mode == 'deterministic':
            self.getIntp = GetIntp_deterministic(model, resolution, exp_size)
        self.register_buffer('lat',model.coor[1], persistent=False)
        self.register_buffer('mask', model.land_mask, persistent=False)
        self.register_buffer('iy', model.iy, persistent=False)
        self.res = resolution

    def forward(self, interpolated_field, grid_field, motion):  # in * 193 * 505
        y_degree = (motion[:, 1:2] / self.iy) * self.res
        f_itpl = 2 * 7.292e-5 * torch.sin((self.lat + y_degree) * torch.pi / 180)
        f_grid = 2 * 7.292e-5 * torch.sin(self.lat * torch.pi / 180)
        interpolating_field = self.getIntp(interpolated_field, motion)

        # q_last = (curl_last + f_last) * torch.cos((self.lat + y_degree) * torch.pi / 180)
        q_interpolating = interpolating_field + f_itpl
        q_grid = grid_field + f_grid
        delta_q = q_interpolating - q_grid
        # return delta_q[..., ~self.mask], delta_q.abs()
        return delta_q[..., ~self.mask].abs()  # B, 1, N


class PhysicsLoss(nn.Module):
    """
    Physics-guided Loss: combination of PV conservation and Deformation Constraint
    """

    def __init__(self, config, model):
        super(PhysicsLoss, self).__init__()
        self.register_buffer('mask', model.land_mask, persistent=False)
        self.register_buffer('ix', model.ix, persistent=False)
        self.register_buffer('iy', model.iy, persistent=False)
        self.register_buffer('norm_w', model.norm_w, persistent=False)
        self.eps = config.expand_size
        self.res = config.resolution
        self.rk = config.rk_mode

        self.dp_para = DeltaPV(model, self.res, self.eps, 'parallel')
        self.dp_dete = DeltaPV(model, self.res, self.eps, 'deterministic')
        # self.scale = 1.6  # 1.6: 0.2; 2: 0.135, 3: e^-3 = 1/20, e^-4 = 1/55  e^-5 = 1/148
        self.scale = math.exp(0.2) - math.exp(1)
        self.S_PERC_98 = 4.178e-05
        self.DQ_PERC_98 = 2.5011e-05

        '''Sigma (Deformation):
          -> 95th Percentile: 2.998500e-05  
          -> 98th Percentile: 4.178000e-05                            
          Delta Q (PV residual):  
          -> 95th Percentile: 1.548000e-05  
          -> 98th Percentile: 2.501100e-05                    
          Lamda (Sigma/DQ):
          -> 95th Percentile: 1.549500e+01
          -> 98th Percentile: 2.499750e+01'''

        # self.Pearson = LocalPearsonLoss(self.mask, 36, 18, 200)  # 36, 18, 200
        # RK module
        if '1' in self.rk:
            self.RK_para = RK1(model)
            self.RK_dete = RK1(model)
        elif '2' in self.rk:
            self.RK_para = RK2(model, self.res, self.eps, 'parallel')
            self.RK_dete = RK2(model, self.res, self.eps, 'deterministic')
        elif '3' in self.rk:
            self.RK_para = RK3(model, self.res, self.eps, 'parallel')
            self.RK_dete = RK3(model, self.res, self.eps, 'deterministic')
        elif '4' in self.rk:
            self.RK_para = RK4(model, self.res, self.eps, 'parallel')
            self.RK_dete = RK4(model, self.res, self.eps, 'deterministic')

    def forward(self, data, pred, target):
        """
        Args:
            data: input sequence [batch_size, 1*seq_len, lat, lon]
            pred: prediction [batch_size, 1, lat, lon]
            target: target [batch_size, 2, sea_points] (ocean points only)
        """
        dt = 86400.  # seconds in one day
        uv_last = data[:, :2, :, :]
        uv_last[..., self.mask] = 0
        pred_clone = pred.clone()
        pred_clone[..., self.mask] = 0

        # calc grad
        u_x, u_y, v_x, v_y = calc_grad_4order(uv_last, self.ix, self.iy, self.eps)
        curl_last = (v_x - u_y) * 0.001
        s_last = torch.sqrt((u_x - v_y) ** 2 + (v_x + u_y) ** 2 + 1e-12) * 0.001
        del u_x, u_y, v_x, v_y
        u_x, u_y, v_x, v_y = calc_grad_4order(pred_clone, self.ix, self.iy, self.eps)
        curl_pred = (v_x - u_y) * 0.001
        s_pred = torch.sqrt((u_x - v_y) ** 2 + (v_x + u_y) ** 2 + 1e-12) * 0.001
        del u_x, u_y, v_x, v_y

        bk_motion = self.RK_para(uv_last, uv_last, -dt, torch.zeros_like(uv_last))  # km
        fw_motion = self.RK_para(pred_clone, pred, dt, torch.zeros_like(pred))  # km

        # lat weights, NEW MSE latw
        pred_sea = pred[:, :, ~self.mask]
        mse_sea = (pred_sea - target) ** 2 * self.norm_w
        lat_w_mse = mse_sea.sum(dim=-1).mean()

        dq1 = self.dp_para(curl_last, curl_pred, bk_motion.detach())
        dq2 = self.dp_para(curl_pred, curl_last, fw_motion.detach())
        # sum_dq = dq1 + dq2

        # amount norm
        # sigma_add = s_pred + s_last
        # sum_s = sigma_add[..., ~self.mask]
        # s_norm = torch.clamp(sum_s / self.S_PERC_98, max=1.0) * self.scale  # B, 1, N
        # weight_s = torch.exp(-s_norm.detach()) * self.norm_w  # [B, 1, N]
        # w_s = weight_s

        # dq_norm = torch.clamp(sum_dq / self.DQ_PERC_98, max=1.0) * self.scale  # B, 1, N
        # weight_dq = torch.exp(-dq_norm.detach()) * self.norm_w  # [B, 1, N]
        # w_dq = weight_dq

        # sigma_add = s_pred + s_last
        # sum_s = sigma_add[..., ~self.mask]
        # s_norm = torch.clamp(sum_s / self.S_PERC_98, max=1.0) * self.scale  # B, 1, N
        # weight_s = torch.log(s_norm.detach() + math.exp(1)) * self.norm_w  # [B, 1, N]
        # w_s = weight_s

        # dq_norm = torch.clamp(sum_dq / self.DQ_PERC_98, max=1.0) * self.scale  # B, 1, N
        # weight_dq = torch.log(dq_norm.detach() + math.exp(1)) * self.norm_w  # [B, 1, N]
        # w_dq = weight_dq

        # Potential Vorticity (PV Residual, dq) constraint
        delta_q = 0.5 * (dq1 ** 2 + dq2 ** 2) * self.norm_w  # w_s.detach()  self.norm_w
        del dq1, dq2 # weight_s, weight_dq
        lat_w_dq = delta_q.sum(dim=-1).mean()

        # fluid strain deformation (S) constraint
        sigma_sea = s_pred[..., ~self.mask]  # B, 1, N
        sigma_sqr = sigma_sea ** 2 * self.norm_w  # w_dq.detach()  self.norm_w
        lat_w_sigma = sigma_sqr.sum(dim=-1).mean()

        # Pearson
        # lat_w_pearson = self.Pearson(sigma_last + sigma_pred, dq_forward + dq_backward)
        # # sigma curl
        # sigma_sea = sigma_pred[:, :, ~self.mask]
        # curl_sea = curl_pred[:, :, ~self.mask]
        # relu_sqr = F.relu(torch.abs(sigma_sea) - torch.abs(curl_sea * 0.8)) ** 2 * self.norm_w
        # lat_w_relu = relu_sqr.sum(dim=-1).mean()

        del uv_last, pred_clone, curl_last, curl_pred, s_pred
        del bk_motion, fw_motion, delta_q, sigma_sqr, sigma_sea

        return {
            'mse': lat_w_mse,
            'pv': lat_w_dq,
            'sigma': lat_w_sigma,
            'pearson': 0.
        }
