"""Coarse-grain the global energy spectrum and save daily results to disk.

For each date and each requested dataset, the global area-weighted mean of the
coarse kinetic energy E_c(ell) is computed over 20 logarithmic scales
(20 to 1200 km):

  plot_file/coarse_daily/{dataset}/coarse_{date}.npz

Each npz stores only the scales and Ec arrays. Existing files are skipped, so the
script can be resumed across multiple runs. The energy flux Pi is handled
separately (see coarse_maps.py).
"""

import os
import argparse
import numpy as np

from common import (producing_lonlat, load_uv, yyyymmdd_range, set_lead,
                    lead_subdir)
from coarse import make_ocean_mask, coarse_grain_uv_flux, area_mean

START, END = '20190101', '20191231'
EVERY_DAYS = 1
N_SCALES = 20
L_MIN_KM, L_MAX_KM = 20.0, 1200.0

NAMES = ['data', 'phy']

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_DATA_DIR = os.path.join(HERE, 'plot_file')
DAILY_DIR = os.path.join(PLOT_DATA_DIR, 'coarse_daily')


def process_one(name, date_str, mask, lat, scales, lead):
    """Compute and cache the (dataset, date) result; skip if the file exists."""
    d = os.path.join(DAILY_DIR, lead_subdir(name, lead))
    os.makedirs(d, exist_ok=True)
    fpath = os.path.join(d, f'coarse_{date_str}.npz')
    if os.path.exists(fpath):
        return
    uv = load_uv(name, date_str)
    if uv is None:
        return
    u, v = uv
    u = u.astype(np.float32); v = v.astype(np.float32)
    u[~mask] = 0.0; v[~mask] = 0.0
    Ec = np.zeros(N_SCALES)
    for si, Lkm in enumerate(scales):
        _, _, Ecf, _ = coarse_grain_uv_flux(u, v, mask, Lkm, lat, compute_flux=False)
        Ec[si] = area_mean(Ecf, mask, lat)
    np.savez(fpath, scales=scales, Ec=Ec)


def main():
    ap = argparse.ArgumentParser(description="Coarse-grain the energy spectrum and save daily results")
    ap.add_argument('--lead', type=int, default=10, help='Lead time (days)')
    ap.add_argument('--name', default=None, choices=NAMES,
                    help='Process a single dataset (data/phy); defaults to all')
    ap.add_argument('--start', default=START)
    ap.add_argument('--end', default=END)
    ap.add_argument('--every', type=int, default=EVERY_DAYS)
    args = ap.parse_args()

    set_lead(args.lead)

    os.makedirs(DAILY_DIR, exist_ok=True)

    lat, _ = producing_lonlat()
    mask = make_ocean_mask(lat)
    scales = np.logspace(np.log10(L_MIN_KM), np.log10(L_MAX_KM), N_SCALES)

    dates = list(yyyymmdd_range(args.start, args.end))[::args.every]
    names = [args.name] if args.name else NAMES
    print(f"Dates: {len(dates)}, scales: {len(scales)}, datasets: {names}")
    for di, date_str in enumerate(dates):
        for n in names:
            process_one(n, date_str, mask, lat, scales, args.lead)
        if (di + 1) % 5 == 0 or di == len(dates) - 1:
            print(f"  Processed {di + 1}/{len(dates)} days")

    print("Done (E_c saved to coarse_daily).")


if __name__ == '__main__':
    main()
