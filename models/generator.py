import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=256, channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)