"""Coarse-graining utilities following Aluie's filtering-scale theory (see FlowSieve).

Reference: Storer et al., "Satellite altimetry reveals spatially nonlocal kinetic
energy cascade in the global ocean".

  - A Gaussian low-pass filter of width ell is applied to the surface velocity,
    giving the coarse field u_ell = G_ell * u.
  - Coarse kinetic energy E_c(x, ell) = 0.5 |u_ell|^2.
  - The scale-to-scale energy flux is Pi(x, ell) = -tau_ij S_ij, where the subgrid
    stress is tau_ij = overline(u_i u_j) - u_i u_j and the coarse strain rate is
    S_ij = 0.5 (d u_i / d x_j + d u_j / d x_i). Positive Pi is a forward cascade
    (energy from large to small scales), negative Pi is an inverse cascade.
  - Land-sea boundaries use normalized convolution: u_ell = G_ell*(u*M) / G_ell*M,
    where M is the ocean mask (1 = ocean).

Gaussian kernel convention (Aluie): G_ell(r) = exp(-r^2 / (2 ell^2)) / (2 pi ell^2),
i.e. sigma = ell. On the sphere the longitude grid spacing shrinks as cos(phi), so
the longitude sigma is scaled by 1/cos(phi); the latitude sigma is constant.

Memory: all computations run in float32, the filter is split into two batches, and
the mask weights are reused to avoid the peak memory of 2040x4320 float64 fields.
"""

import numpy as np
from scipy import ndimage

from common import KM_PER_DEG, RES, LAT_MAX, load_land_mask

PIX_PER_DEG = 1.0 / RES          # = 12 px/deg
_MIN_COSLAT = 0.05               # minimum cos(lat), avoids sigma divergence near poles


def make_ocean_mask(lat):
    """Build the ocean mask (land = False), removing grid rows north of 80 N."""
    land = load_land_mask()
    ocean = ~land
    ocean[lat > LAT_MAX, :] = False
    return ocean


def _sigma_params(scale_km, lat):
    """Return (sigma_lat in px, sx_arr per-row longitude sigma in px)."""
    sigma_deg = scale_km / KM_PER_DEG
    sigma_lat = sigma_deg * PIX_PER_DEG
    coslat = np.maximum(np.cos(np.deg2rad(lat)), _MIN_COSLAT)
    sx_arr = sigma_lat / coslat
    return sigma_lat, sx_arr


def _compute_den(mask, sigma_lat, sx_arr):
    """Compute the normalization denominator G_ell * M (float32), reused across fields."""
    ny = mask.shape[0]
    den = ndimage.gaussian_filter1d(mask.astype(np.float32), sigma_lat,
                                    axis=0, mode='nearest')
    for j in range(ny):
        den[j, :] = ndimage.gaussian_filter1d(den[j, :], sx_arr[j], mode='wrap')
    return den


def _apply_filter(field, den, sigma_lat, sx_arr):
    """Apply the spherical Gaussian low-pass filter to field (K, nlat, nlon) and divide by den.

    Points outside the mask are NaN in the output.
    """
    ny = field.shape[1]
    num = ndimage.gaussian_filter1d(field, sigma_lat, axis=1, mode='nearest')
    for j in range(ny):
        num[:, j, :] = ndimage.gaussian_filter1d(num[:, j, :], sx_arr[j],
                                                 axis=-1, mode='wrap')
    den_b = den[None, :, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        out = num / den_b
    out = np.where(den_b < 1e-12, np.nan, out)
    return out


def _gradients_f32(u, v, lat):
    """Fourth-order central differences of velocity (float32), periodic in longitude."""
    ny, nx = u.shape
    dlon_m = ((KM_PER_DEG * RES * np.cos(np.deg2rad(lat)))[:, None] * 1000.0).astype(np.float32)
    dlat_m = np.float32(KM_PER_DEG * RES * 1000.0)

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


def coarse_grain_uv_flux(u, v, mask, scale_km, lat, compute_flux=True):
    """Coarse-grain u and v, returning (u_c, v_c, E_c, Pi).

    Units: velocity m/s, energy m^2/s^2, flux m^2/s^3. u and v must be 0 on land.
    With compute_flux=False the subgrid stress and gradients are skipped (Pi=None).
    """
    sigma_lat, sx_arr = _sigma_params(scale_km, lat)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    den = _compute_den(mask, sigma_lat, sx_arr)

    # Batch 1: u and v.
    f1 = _apply_filter(np.stack([u, v]), den, sigma_lat, sx_arr)
    u_c = f1[0]
    v_c = f1[1]
    del f1

    E_c = np.float32(0.5) * (u_c * u_c + v_c * v_c)

    if not compute_flux:
        return u_c, v_c, E_c, None

    # Batch 2: uu, uv, vv (subgrid stress needs product-then-filter).
    f2 = _apply_filter(np.stack([u * u, u * v, v * v]), den, sigma_lat, sx_arr)
    uu_c = f2[0]
    uv_c = f2[1]
    vv_c = f2[2]
    del f2

    # Subgrid stress.
    tau_xx = uu_c - u_c * u_c
    tau_xy = uv_c - u_c * v_c
    tau_yy = vv_c - v_c * v_c

    # Coarse strain rate (land is set to 0 before taking gradients).
    u_cf = np.where(np.isfinite(u_c), u_c, np.float32(0.0))
    v_cf = np.where(np.isfinite(v_c), v_c, np.float32(0.0))
    du_dx, du_dy, dv_dx, dv_dy = _gradients_f32(u_cf, v_cf, lat)
    del u_cf, v_cf
    S_xx = du_dx
    S_xy = np.float32(0.5) * (du_dy + dv_dx)
    S_yy = dv_dy
    del du_dx, du_dy, dv_dx, dv_dy

    Pi = -(tau_xx * S_xx + np.float32(2.0) * tau_xy * S_xy + tau_yy * S_yy)
    Pi = np.where(np.isfinite(Pi), Pi, np.nan)

    return u_c, v_c, E_c, Pi


def area_mean(field, mask, lat):
    """Area-weighted global mean (cos-latitude weights), over finite masked values only."""
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], field.shape)
    m = mask & np.isfinite(field)
    if not m.any():
        return np.nan
    return float(np.sum(w[m] * field[m]) / np.sum(w[m]))


def coarse_spectrum(scale_km_arr, Ec_arr):
    """Numerically differentiate the global coarse KE E_c to get the spectral density E(k_ell).

    Returns (L_center in km, Ek). E(k) = dE_bar/dk with k = 1/ell (ell in meters).
    """
    L = np.asarray(scale_km_arr, dtype=np.float64)
    E = np.asarray(Ec_arr, dtype=np.float64)
    order = np.argsort(L)
    L, E = L[order], E[order]
    k = 1.0 / (L * 1000.0)
    kc = np.sqrt(k[1:] * k[:-1])
    Lc = 1.0 / (kc * 1000.0)
    Ek = np.diff(E) / np.diff(k)
    return Lc, Ek


# --------------------------------------------------------------------------- #
# float16 quantization for daily Pi fields (saves about half the space)
# --------------------------------------------------------------------------- #

PI_SCALE = 1e9       # typical Pi (m^2/s^3) magnitude is 1e-9 to 1e-5
PI_CLIP = 60000.0    # float16 safe upper bound


def save_pi_f16(path, Pi):
    """Store Pi scaled by PI_SCALE as float16 (NaN preserved)."""
    v = np.asarray(Pi, dtype=np.float32) * PI_SCALE
    v = np.clip(v, -PI_CLIP, PI_CLIP)
    np.save(path, v.astype(np.float16))


def load_pi_f16(path):
    """Load a float16 Pi field and restore float32 (m^2/s^3)."""
    return np.load(path).astype(np.float32) / PI_SCALE
