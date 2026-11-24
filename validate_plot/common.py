"""Shared utilities for the validate tasks: data loading, coordinates, land mask,
climatology, vorticity, and spectral computation.

Data conventions (consistent with instruct.py):
  - Grid 2040 x 4320, 1/12 degree, lat = -79.9167 : 1/12 : 90.0, lon = -180 : 1/12 : 179.9167
  - label / wenhai are originally 2041 rows; slice [..., 1:, :] to get 2040 rows
  - data / phy are already 2040 rows; land is unmasked, so NaN out land via land_mask
"""
import os
from datetime import datetime, timedelta

import numpy as np

R_EARTH = 6371.0                                   # km
RES = 1.0 / 12.0                                   # 1/12 degree
KM_PER_DEG = 2.0 * np.pi * R_EARTH / 360.0         # ~111.19 km/deg
NLAT, NLON = 2040, 4320

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = './source'
LAND_MASK_PATH = os.path.join(SOURCE_DIR, 'land_mask.npy')
CLIM_UV_PATH = os.path.join(SOURCE_DIR, 'Global_uv_clim_93_13.npy')
CLIM_EKE_PATH = os.path.join(SOURCE_DIR, 'Global_eke_clim_93_13.npy')
UV_MEAN_PATH = os.path.join(SOURCE_DIR, 'Global_uv_mean_93_13.npy')

# NOTE: the paths below are example paths from the original Linux cluster;
# edit them to point to your own dataset before running.
DATASETS = {
    'label':  dict(vel_dir='./OSC', pattern='uv0_{date}.npy', fmt='npy'),
    'data':   dict(vel_dir='./eddy_identify/eddy_in', pattern='{date}_uv0_L1_10d.npy', fmt='npy'),
    'phy':    dict(vel_dir='./eddy_identify/eddy_in', pattern='{date}_uv0_L123_93_13_1_5_0_10d.npy', fmt='npy'),
    'wenhai': dict(vel_dir='./Wenhai_pred/uv', pattern='{date}_pred_wenhai_uv_10d.npz', fmt='npz'),
}


def set_lead(lead):
    """Update the data/phy/wenhai paths for a given forecast lead time `lead`
    (label is unchanged). Edit the vel_dir/pattern values below to match your own
    dataset layout; `lead` selects the forecast lead time used in the file naming."""
    DATASETS['data'] = dict(
        vel_dir='./eddy_identify/eddy_in',
        pattern=f'{{date}}_uv0_L1_{lead}d.npy', fmt='npy')
    DATASETS['phy'] = dict(
        vel_dir='./eddy_identify/eddy_in',
        pattern=f'{{date}}_uv0_L123_93_13_1_5_0_{lead}d.npy', fmt='npy')
    DATASETS['wenhai'] = dict(
        vel_dir=f'./Wenhai_pred/uv',
        pattern=f'{{date}}_pred_wenhai_uv_{lead}d.npz', fmt='npz')


def lead_subdir(name, lead):
    """Subdirectory name: label stays as `name` (shared across leads); data/phy get
    a `_lead{lead}` suffix when lead != 10.

    lead10 keeps the original name for compatibility with existing data dirs.
    """
    if name == 'label':
        return name
    return name if lead == 10 else f'{name}_lead{lead}'


def lead_suffix(lead):
    """File/output name suffix: lead10 keeps the original name (backward compatible),
    other leads get a `_lead{lead}` suffix."""
    return '' if lead == 10 else f'_lead{lead}'


def producing_lonlat():
    """Generate the 2040x4320 latitude/longitude grids (in degrees)."""
    lat = -80.0 + RES + RES * np.arange(NLAT)
    lon = -180.0 + RES * np.arange(NLON)
    return lat, lon


def yyyymmdd_range(start, end):
    d0 = datetime.strptime(start, "%Y%m%d")
    d1 = datetime.strptime(end, "%Y%m%d")
    while d0 <= d1:
        yield d0.strftime("%Y%m%d")
        d0 += timedelta(days=1)


def load_land_mask():
    """Load the land-sea mask (land = True), cropped to (2040, 4320)."""
    lm = np.load(LAND_MASK_PATH).astype(bool)
    if lm.shape[0] != NLAT:
        lm = lm[1:, :]
    return lm


def load_clim_uv():
    """Load the 1993-2013 u/v climatology, returning (u_clim, v_clim) with shape (2040, 4320)."""
    clim = np.load(CLIM_UV_PATH)
    clim = np.asarray(clim, dtype=np.float64)
    if clim.shape[-2] != NLAT:
        clim = clim[..., 1:, :]
    return clim[0], clim[1]


_uv_mean_cache = None


def _load_uv_mean():
    """Load the (u_mean, v_mean) global means from Global_uv_mean_93_13.npy (cached)."""
    global _uv_mean_cache
    if _uv_mean_cache is None:
        m = np.load(UV_MEAN_PATH)
        _uv_mean_cache = (float(m[0]), float(m[1]))
    return _uv_mean_cache


def load_uv(data_key, date_str):
    """Load (u, v) for a given dataset and date, shape (2040, 4320). Returns None if
    the file does not exist.

    Extreme values (non-NaN, abs > 10 m/s) are replaced by the global mean (outlier cleaning).
    """
    cfg = DATASETS[data_key]
    pth = os.path.join(cfg['vel_dir'], cfg['pattern'].format(date=date_str))
    if not os.path.exists(pth):
        return None
    if cfg['fmt'] == 'npy':
        arr = np.asarray(np.load(pth).squeeze(), dtype=np.float64)
    else:
        arr = np.asarray(np.load(pth)['data'].squeeze(), dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 3:        # [zos, u, v] -> [u, v]
        arr = arr[1:3]
    if arr.shape[-2] != NLAT:                       # 2041 -> 2040
        arr = arr[..., 1:, :]
    u, v = arr[0], arr[1]
    u_mean, v_mean = _load_uv_mean()
    u = np.where((~np.isnan(u)) & (np.abs(u) > 10.0), u_mean, u)
    v = np.where((~np.isnan(v)) & (np.abs(v) > 10.0), v_mean, v)
    return u, v


def latitude_weights(lat):
    """Latitude weights cos(lat), shape (ny,)."""
    return np.cos(np.deg2rad(lat))


def compute_gradients(u, v, lat):
    """Compute velocity gradients (du_dx, du_dy, dv_dx, dv_dy) via 4th-order central
    differences, units s^-1.

    x (longitude) is periodic; y (latitude) is non-periodic (2nd-order one-sided at boundaries)."""
    ny, nx = u.shape
    dlon_m = (KM_PER_DEG * RES * np.cos(np.deg2rad(lat)))[:, None] * 1000.0   # (ny,1) m
    dlat_m = KM_PER_DEG * RES * 1000.0                                        # m

    def ddx(f):
        return (np.roll(f, 2, axis=1) - 8.0 * np.roll(f, 1, axis=1)
                + 8.0 * np.roll(f, -1, axis=1) - np.roll(f, -2, axis=1)) / (12.0 * dlon_m)

    def ddy(f):
        g = np.zeros_like(f)
        g[2:-2, :] = (f[:-4, :] - 8.0 * f[1:-3, :] + 8.0 * f[3:-1, :] - f[4:, :]) / (12.0 * dlat_m)
        g[0, :] = (-3.0 * f[0, :] + 4.0 * f[1, :] - f[2, :]) / (2.0 * dlat_m)
        g[1, :] = (f[2, :] - f[0, :]) / (2.0 * dlat_m)
        g[-2, :] = (f[-1, :] - f[-3, :]) / (2.0 * dlat_m)
        g[-1, :] = (3.0 * f[-1, :] - 4.0 * f[-2, :] + f[-3, :]) / (2.0 * dlat_m)
        return g

    return ddx(u), ddy(u), ddx(v), ddy(v)


def compute_vorticity(u, v, lat):
    """Vorticity omega = dv/dx - du/dy, units s^-1."""
    _, du_dy, dv_dx, _ = compute_gradients(u, v, lat)
    return dv_dx - du_dy


def compute_divergence(u, v, lat):
    """Divergence delta = du/dx + dv/dy, units s^-1."""
    du_dx, _, _, dv_dy = compute_gradients(u, v, lat)
    return du_dx + dv_dy


def trim_moments(x, clip_frac):
    """Compute (skew, kurt, std) after trimming (dropping) the extreme tails."""
    if clip_frac > 0:
        lo = np.percentile(x, 100 * clip_frac / 2)
        hi = np.percentile(x, 100 * (1 - clip_frac / 2))
        x = x[(x >= lo) & (x <= hi)]
    mu, sd = x.mean(), x.std()
    skew = float(np.mean((x - mu) ** 3) / sd ** 3)
    kurt = float(np.mean((x - mu) ** 4) / sd ** 4)
    return skew, kurt, float(sd)


def trim_normalize(x, clip_frac):
    """Trim (drop) the extreme tails, then normalize by the std of the remaining values
    (used for PDF plotting)."""
    if clip_frac > 0:
        lo = np.percentile(x, 100 * clip_frac / 2)
        hi = np.percentile(x, 100 * (1 - clip_frac / 2))
        x = x[(x >= lo) & (x <= hi)]
    sd = x.std()
    return x / sd if sd > 0 else x


def trim_mean_std(x, clip_frac):
    """Compute (mean, std) after trimming (dropping) the extreme tails; no subsampling."""
    if clip_frac > 0:
        lo = np.percentile(x, 100 * clip_frac / 2)
        hi = np.percentile(x, 100 * (1 - clip_frac / 2))
        x = x[(x >= lo) & (x <= hi)]
    return float(x.mean()), float(x.std())


LAT_MAX = 80.0   # upper bound for the statistics domain: keep -80S ~ 80N only (exclude Arctic extremes)


def mask_high_lat(field, lat):
    """Set rows with latitude > LAT_MAX (above 80N) to NaN."""
    field = np.asarray(field).copy()
    field[np.asarray(lat) > LAT_MAX, :] = np.nan
    return field


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def weighted_rmse(pred, label, w):
    """Latitude-weighted RMSE (ignores NaN, i.e. ocean points only). w must broadcast to (ny, nx)."""
    mask = ~np.isnan(pred) & ~np.isnan(label)
    if not mask.any():
        return np.nan
    diff = pred[mask] - label[mask]
    ww = np.broadcast_to(w, pred.shape)[mask]
    return float(np.sqrt(np.sum(ww * diff ** 2) / np.sum(ww)))


def weighted_acc(pred, label, w):
    """Latitude-weighted spatial anomaly correlation coefficient (anomalies after
    removing the spatially weighted mean)."""
    mask = ~np.isnan(pred) & ~np.isnan(label)
    if mask.sum() < 3:
        return np.nan
    p, l = pred[mask], label[mask]
    ww = np.broadcast_to(w, pred.shape)[mask]
    pm = np.sum(ww * p) / np.sum(ww)
    lm = np.sum(ww * l) / np.sum(ww)
    pa, la = p - pm, l - lm
    num = np.sum(ww * pa * la)
    den = np.sqrt(np.sum(ww * pa ** 2) * np.sum(ww * la ** 2))
    return float(num / den) if den > 0 else np.nan


# --------------------------------------------------------------------------- #
# Spectral computation
# --------------------------------------------------------------------------- #

def _detrend_and_taper(field):
    """Remove the spatial mean, fill land with 0, apply a 2D Hann window, and return
    the windowed field together with the power correction factor."""
    f = field - np.nanmean(field)
    f = np.where(np.isnan(f), 0.0, f)
    ny, nx = f.shape
    w = np.outer(np.hanning(ny), np.hanning(nx))
    return f * w, 1.0 / np.mean(w ** 2)


def radial_wavenumbers(nx, ny):
    """Return (kx, ky, k_h) in cycles/km (equator reference)."""
    kx = np.fft.fftfreq(nx, d=RES) / KM_PER_DEG      # cycles/km
    ky = np.fft.fftfreq(ny, d=RES) / KM_PER_DEG
    KX, KY = np.meshgrid(kx, ky)
    return kx, ky, np.sqrt(KX ** 2 + KY ** 2)


def azimuthal_average(power2d, k_h, nbins=80, kmin=None, kmax=None):
    """Azimuthal average over wavenumber rings, returning (k_center, mean_power).
    power2d and k_h must have the same shape."""
    kf = k_h.ravel()
    pf = power2d.ravel()
    valid = (kf > 0) & np.isfinite(pf)
    kf, pf = kf[valid], pf[valid]
    if kmin is None:
        kmin = kf.min()
    if kmax is None:
        kmax = kf.max()
    edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    idx = np.digitize(kf, edges)
    k_center = np.sqrt(edges[1:] * edges[:-1])
    mean = np.full(nbins, np.nan)
    for i in range(1, nbins + 1):
        sel = idx == i
        if sel.any():
            mean[i - 1] = pf[sel].mean()
    return k_center, mean


def fit_slope(k, power, k_lo=None, k_hi=None):
    """Fit the slope of log10(P) vs log10(k) over [k_lo, k_hi]. k is in cycles/km."""
    m = np.isfinite(power) & (power > 0) & (k > 0)
    if k_lo is not None:
        m &= (k >= k_lo)
    if k_hi is not None:
        m &= (k <= k_hi)
    if m.sum() < 5:
        return np.nan
    x = np.log10(k[m])
    y = np.log10(power[m])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)
