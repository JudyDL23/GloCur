import torch
import torch.nn as nn
from model.LossFunc import global_expand
# t is the num of hours ever since 1950,1,1


def max_min_norm(data):
    min_val = data.max()
    max_val = data.min()
    return (data - min_val) / (max_val - min_val + 1e-8)


class SpatioTemporalConv(nn.Module):
    def __init__(self, land_mask, seabed, coor):
        super().__init__()
        self.register_buffer('land', land_mask)
        self.register_buffer('ocean', ~land_mask)
        self.register_buffer('depth', max_min_norm(-1. * seabed))  # H * W
        self.register_buffer('pos_sin_emb', torch.sin(coor * torch.pi/180.))  # 2 * H * W
        self.register_buffer('pos_cos_emb', torch.cos(coor * torch.pi/180.))
        print(self.pos_sin_emb.shape)     # 2,H,W
        print(self.pos_cos_emb.shape)     # 2,H,W
        print(self.depth.shape)  # 1,H,W
        print(self.ocean.shape) # 1,H,W
        print(self.land.shape)
        fixed_feat = [
            self.pos_sin_emb,     # 2,H,W
            self.pos_cos_emb,     # 2,H,W
            self.depth.unsqueeze(0),  # 1,H,W
            self.ocean.float().unsqueeze(0), # 1,H,W
            self.land.float().unsqueeze(0),  # 1,H,W
        ]
        self.register_buffer('fixed_feat', torch.cat(fixed_feat, dim=0))  # (7, H, W)

        # Network architecture unchanged
        self.conv3d = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, t):
        B = t.shape[0]
        H, W = self.land.shape
        t = t.to(self.fixed_feat.device)

        t = t.view(B, 1, 1, 1)
        t_emb = 2 * torch.pi * t / (24. * 365.24)
        t_sin = torch.sin(t_emb).expand(B, 1, H, W)
        t_cos = torch.cos(t_emb).expand(B, 1, H, W)

        # fixed_feat: (7, H, W) -> repeat directly to the batch dimension
        x = torch.cat([
            t_sin,
            t_cos,
            self.fixed_feat.unsqueeze(0).repeat(B, 1, 1, 1)  # expand only once here
        ], dim=1)  # final shape B, 9, H, W
        del t_sin, t_cos, t_emb

        x = global_expand(x, 3)
        x = self.conv3d(x)
        
        return x[..., :-3, 3:-3]