from torch import nn

from ciflows.distributions.pgm import LinearGaussianDag
from ciflows.reduction.resnetvae import DeepResNetDecoder, ResNetEncoder


class ResnetFreeformflow(nn.Module):
    def __init__(
        self,
        latent: LinearGaussianDag,
        latent_dim: int,
        num_blocks_per_stage=3,
    ) -> None:
        super(ResnetFreeformflow, self).__init__()

        self.encoder = ResNetEncoder(latent_dim)  # Same encoder as before
        self.decoder = DeepResNetDecoder(latent_dim, num_blocks_per_stage)
        self.latent = latent
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples, distr_idx=None):
        z = self.latent.sample(num_samples, distr_idx=distr_idx)
        return self.decoder(z), z
