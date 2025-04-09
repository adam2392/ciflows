import torch
import torch.nn as nn
import torch.nn.functional as F


class ActNorm1D(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.logs = nn.Parameter(torch.zeros(1, num_features))

    def forward(self, x, reverse=False):
        if not reverse:
            x = (x + self.bias) * torch.exp(self.logs)
            return x, 0
        else:
            x = x * torch.exp(-self.logs) - self.bias
            return x, 0


class InvertibleLinear1D(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        w_init = torch.qr(torch.randn(num_features, num_features))[0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, reverse=False):
        if not reverse:
            x = x @ self.weight
            return x, 0
        else:
            x = x @ torch.inverse(self.weight)
            return x, 0


class AffineCoupling1D(nn.Module):
    def __init__(self, num_channels, hidden_channels):
        super().__init__()
        assert num_channels % 2 == 0, "num_channels must be even for coupling."
        self.half = num_channels // 2
        self.net = nn.Sequential(
            nn.Linear(self.half, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, self.half * 2),
        )

    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.half], x[:, self.half:]
        h = self.net(x1)
        shift, log_scale = h.chunk(2, dim=1)
        scale = torch.exp(log_scale)
        if not reverse:
            x2 = x2 * scale + shift
        else:
            x2 = (x2 - shift) / scale
        x = torch.cat([x1, x2], dim=1)
        return x, 0


class GlowStep1D(nn.Module):
    def __init__(self, num_channels, hidden_channels):
        super().__init__()
        self.actnorm = ActNorm1D(num_channels)
        self.inv_linear = InvertibleLinear1D(num_channels)
        self.coupling = AffineCoupling1D(num_channels, hidden_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            x, _ = self.actnorm(x)
            x, _ = self.inv_linear(x)
            x, _ = self.coupling(x)
        else:
            x, _ = self.coupling(x, reverse=True)
            x, _ = self.inv_linear(x, reverse=True)
            x, _ = self.actnorm(x, reverse=True)
        return x, 0


class GlowLevel1D(nn.Module):
    def __init__(self, num_channels, hidden_channels, num_steps):
        super().__init__()
        self.steps = nn.ModuleList([
            GlowStep1D(num_channels, hidden_channels)
            for _ in range(num_steps)
        ])

    def forward(self, x, reverse=False):
        if not reverse:
            for step in self.steps:
                x, _ = step(x, reverse=False)
        else:
            for step in reversed(self.steps):
                x, _ = step(x, reverse=True)
        return x, 0


class MultiScaleGlow1D(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_levels=3, steps_per_level=4):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels must be even"
        self.input_proj = nn.Linear(in_channels, in_channels)  # identity by default
        self.levels = nn.ModuleList()
        channels = in_channels
        for _ in range(num_levels):
            self.levels.append(GlowLevel1D(channels, hidden_channels, steps_per_level))

    def forward(self, x, reverse=False):
        x = self.input_proj(x)
        if not reverse:
            for level in self.levels:
                x, _ = level(x, reverse=False)
        else:
            for level in reversed(self.levels):
                x, _ = level(x, reverse=True)
        return x


# --- Verification ---
def verify_multiscale_glow_1d():
    batch_size = 32
    in_features = 32
    x = torch.randn(batch_size, in_features)

    model = MultiScaleGlow1D(in_channels=in_features, hidden_channels=64, num_levels=3, steps_per_level=4)
    with torch.no_grad():
        z = model(x)
        x_recon = model(z, reverse=True)
        diff = torch.norm(x - x_recon).item()
        print(f"Reconstruction L2 error: {diff:.6f}")


if __name__ == '__main__':
    print("Verifying Multi-Scale Glow (1D)...")
    verify_multiscale_glow_1d()
