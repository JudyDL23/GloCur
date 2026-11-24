"""Coarse-grain the scale-to-scale energy flux Pi and save daily fields to disk.

For a given filter scale ell (default 1 deg), the daily flux field
Pi(x) = -tau_ij S_ij is computed for the requested datasets and saved as
float16 npy files:

  plot_file/pi_daily/{dataset}/{dataset}_{date}_{scale}deg.npy

Existing files are skipped, so the script can be resumed across multiple runs.

Usage (one scale per run):
  python coarse_maps.py --scale 2.0
  python coarse_maps.py --scale 1.0
  python coarse_maps.py --scale 0.5
"""

import os
import argparse
import numpy as np

from common import producing_lonlat, load_uv, yyyymmdd_range, KM_PER_DEG, set_lead, lead_subdir
from coarse import make_ocean_mask, coarse_grain_uv_flux, save_pi_f16

START, END = '20190101', '20191231'
EVERY_DAYS = 1

NAMES = ['data', 'phy']

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_DATA_DIR = os.path.join(HERE, 'plot_file')
PI_DIR = os.path.join(PLOT_DATA_DIR, 'pi_daily')


def scale_tag(deg):
    return f"{deg:g}".replace('.', 'p')


def main():
    ap = argparse.ArgumentParser(description="Coarse-grain and save daily flux fields")
    ap.add_argument('--scale', type=float, default=1.0, help='Filter scale (deg)')
    ap.add_argument('--name', default=None, choices=NAMES,
                    help='Process a single dataset (data/phy); defaults to all')
    ap.add_argument('--lead', type=int, default=5, help='Lead time (days)')
    ap.add_argument('--start', default=START)
    ap.add_argument('--end', default=END)
    ap.add_argument('--every', type=int, default=EVERY_DAYS)
    args = ap.parse_args()
    set_lead(args.lead)

    scale_deg = args.scale
    scale_km = scale_deg * KM_PER_DEG
    tag = scale_tag(scale_deg)

    os.makedirs(PI_DIR, exist_ok=True)

    lat, _ = producing_lonlat()
    mask = make_ocean_mask(lat)
    dates = list(yyyymmdd_range(args.start, args.end))[::args.every]
    names = [args.name] if args.name else NAMES
    print(f"Scale L={scale_deg:g} deg, datasets: {names}, dates: {len(dates)}")

    # Save daily Pi fields (incremental).
    for n in names:
        os.makedirs(os.path.join(PI_DIR, lead_subdir(n, args.lead)), exist_ok=True)
    for di, date_str in enumerate(dates):
        for n in names:
            fpath = os.path.join(PI_DIR, lead_subdir(n, args.lead), f'{n}_{date_str}_{tag}deg.npy')
            if os.path.exists(fpath):
                continue
            uv = load_uv(n, date_str)
            if uv is None:
                continue
            u, v = uv
            u = u.astype(np.float32); v = v.astype(np.float32)
            u[~mask] = 0.0; v[~mask] = 0.0
            _, _, _, Pi = coarse_grain_uv_flux(u, v, mask, scale_km, lat)
            save_pi_f16(fpath, Pi)
        if (di + 1) % 10 == 0 or di == len(dates) - 1:
            print(f"  Processed {di + 1}/{len(dates)} days")

    print(f"Done ({tag} deg Pi saved to pi_daily).")


if __name__ == '__main__':
    main()
