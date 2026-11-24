# utils/metrics.py
import torch


def check_nan(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN/inf found in {name}: {tensor}")


def calc_acc(pred, true, clim, norm_w, w=True):
    """Compute the anomaly correlation coefficient (ACC)."""
    pred_dev = pred - clim  # [batch_size, c, n_points]
    true_dev = true - clim  # [batch_size, c, n_points]

    if w:
        sqrt_norm_w = torch.sqrt(norm_w)
        weighted_pred = sqrt_norm_w * pred_dev
        weighted_true = sqrt_norm_w * true_dev

        prod_sum = torch.sum(weighted_pred * weighted_true, dim=-1)  # [batch_size, c]
        pred_norm = torch.sqrt(torch.sum(weighted_pred ** 2, dim=-1))  # [batch_size, c]
        true_norm = torch.sqrt(torch.sum(weighted_true ** 2, dim=-1))  # [batch_size, c]
        denominator = pred_norm * true_norm
    else:
        prod_sum = torch.sum(pred_dev * true_dev, dim=-1)  # [batch_size, c]
        pred_norm = torch.sqrt(torch.sum(pred_dev ** 2, dim=-1))  # [batch_size, c]
        true_norm = torch.sqrt(torch.sum(true_dev ** 2, dim=-1))  # [batch_size, c]
        denominator = pred_norm * true_norm

    epsilon = 1e-10
    acc = prod_sum / (denominator + epsilon)  # [batch_size, c]
    return acc


def calc_mse(pred, true, norm_w, w=True):
    """Compute the mean squared error (MSE)."""
    square_error = (pred - true) ** 2  # [batch_size, c, n_points]

    if w:
        weighted_sum = (square_error * norm_w).sum(dim=-1)  # [batch_size, c]
        return weighted_sum  # [batch_size, c]
    else:
        return square_error.mean(dim=-1)  # [batch_size, c]


def calc_eke_mse(uv, target, clim, norm_w, w=True):
    """Compute the eddy kinetic energy mean squared error (EKE MSE)."""
    delta_uv = torch.sum((uv - clim.unsqueeze(0)) ** 2, dim=-2) / 2  # bs * n_points
    delta_target = torch.sum((target - clim.unsqueeze(0)) ** 2, dim=-2) / 2  # bs * n_points
    square_error = (delta_uv - delta_target) ** 2  # [batch_size, n_points]

    if w:
        weighted_mse = (square_error * norm_w).sum(dim=-1)  # [batch_size]
        return weighted_mse  # [batch_size]
    else:
        return square_error.mean(dim=-1)  # [batch_size]


class MetricsCalculator:
    """Metrics calculator, convenient to use during training."""

    def __init__(self, clim, land, norm_w, use_weight=True):
        self.clim = clim[..., ~land]
        self.norm_w = norm_w
        self.use_weight = use_weight
        self.lm = land

    def compute_all(self, pred, target, eke_clim=None):
        # pred: [B, C, n_points], target: [B, C, n_points]
        """Compute all metrics."""
        acc = calc_acc(pred, target, self.clim, self.norm_w, self.use_weight)
        mse = calc_mse(pred, target, self.norm_w, self.use_weight)
        eke_mse = calc_eke_mse(pred, target, self.clim, self.norm_w, self.use_weight)

        if eke_clim is not None:
            if eke_clim.shape[-1] < 10000: 
                eke_clim = eke_clim[..., ~self.lm]
            # TODO: add the eke_mse computation logic here
            pred_eke = torch.sum((pred - self.clim) ** 2, dim=1) * 0.5  # B, C, N ---> # B, N
            targ_eke = torch.sum((target - self.clim) ** 2, dim=1) * 0.5  # B, C, N ---> # B, N
            eke_acc = calc_acc(pred_eke.unsqueeze(-2), targ_eke.unsqueeze(-2),
                               eke_clim.unsqueeze(0), self.norm_w, self.use_weight)  # B, C
        else:
            eke_acc = torch.zeros((pred.shape[0], 1))

        return {
            'acc': acc,
            'mse': mse,
            'eke_mse': eke_mse,  # [c,]
            'eke_acc': eke_acc.squeeze()  # [c,]
        }
