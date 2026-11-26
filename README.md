# Dynamics-preserving deep learning for global ocean surface current forecasts （under review）

This is a physics-guided Swin Transformer for forecasting global ocean surface
currents. The model combines a Swin Transformer backbone with physics-based
constraints, including potential-vorticity (PV) conservation and fluid-deformation
regularization, to improve short-term ocean surface current prediction.

> Note: this repository is provided as a reference implementation while the paper
> is under review. All data paths are relative placeholders that you must configure for your own data.

## Overview

- **Backbone**: Swin Transformer with a encoder-bottleneck-decoder design.
- **Physics constraints**: potential-vorticity conservation and fluid-deformation
  terms are added to the training loss alongside the mean squared error.
- **Data**: GLORYS12 ocean reanalysis on a 1/12 degree global grid.
- **Scale analysis**: Compute cross-scale kinetic-energy flux and coarse-grained energy spectra.

## Repository structure

```
src/
├── model/              # Swin Transformer, physics loss, configuration
│   ├── SwTEarth.py
│   ├── LossFunc.py
│   ├── SpatioTemporalConv.py
│   └── config.py
├── data/               # dataset and preprocessing
│   ├── dataset.py
│   └── preprocessor.py
├── lightning_modules/  # PyTorch Lightning module
│   └── lit_module.py
├── utils/              # metrics, physics, convergence, visualization
├── validate_plot/      # validation and coarse-graining analysis
│   ├── coarse.py
│   ├── coarse_maps.py
│   └── coarse_spectrum.py
└── train.py            # training entry point
```

## Installation

```bash
pip install -r requirements.txt
```

The data paths in `model/config.py` and `validate_plot/common.py` are relative
placeholders (`./data/`, `./source/`, `./ckpt/`, `./params/`). Point them to your
local GLORYS12 data and auxiliary files before training or evaluation.

## Training

```bash
python train.py --loss_mode 1 --seed XX
```

The `--loss_mode` argument is a string of digits. The digit `1` is the mean squared
error. Appending `2` or `3` adds the potential-vorticity or fluid-deformation term
respectively. For example, `--loss_mode 123` trains the full physics-guided model.

## Coarse-graining analysis

The `validate_plot/` directory provides tools for multi-scale energy analysis.

- `coarse.py`: Gaussian coarse-graining, cross-scale energy flux, and energy spectra.
- `coarse_maps.py`: compute and save daily cross-scale flux fields.
- `coarse_spectrum.py`: compute and save daily coarse kinetic energy.

## Reference

The coarse-graining implementation follows:

Storer et al., "Satellite altimetry reveals spatially nonlocal kinetic energy
cascade in the global ocean" (2022).
