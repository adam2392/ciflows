import torch
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, distr_idx=None):
        z = self.encoder(x)

        # treated like a VAE
        _, log_means, log_vars = self.latent.log_prob(
            z, distr_idx=distr_idx.cpu(), return_means_log_vars=True
        )
        embeddings = self.reparameterize(log_means, log_vars)
        recon_x = self.decoder(embeddings)

        # hutchinson_samples = 2
        # surrogate_loss, v_hat, x_hat = volume_change_surrogate(
        #     x,
        #     self.encoder,
        #     self.decoder,
        #     hutchinson_samples=hutchinson_samples,
        # )

        # # get negative log likelihoood over the distributions
        # embed_dim = self.latent_dim
        # v_hat = v_hat.view(-1, embed_dim)
        # loss_nll = (
        #     -self.latent.log_prob(v_hat, distr_idx=distr_idx.cpu()) - surrogate_loss.mean()
        # )

        # compute the log-likelihood of the data
        # distr_idx = distr_idx.cpu()
        # log_prob, log_means, log_vars = self.latent.log_prob(z, distr_idx=distr_idx, return_means_log_vars=True)

        return (
            recon_x,
            log_means,
            log_vars,
        )  # surrogate_loss, loss_nll  # log_prob, log_means, log_vars

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples, distr_idx=None):
        z = self.latent.sample(num_samples, distr_idx=distr_idx)
        return self.decoder(z), z
