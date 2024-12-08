import lightning as pl
import torch
import torch.optim as optim

from .loss import volume_change_surrogate, volume_change_surrogate_transformer


# TODO: finish implementation
class plApproximateFlowModel(pl.LightningModule):
    """Approximate flow model lightning module.

    Here, we define an approximate flow model for training.
    """

    def __init__(
        self,
        latent,
        encoder,
        decoder,
        lr=1e-3,
        lr_min=1e-8,
        lr_scheduler=None,
        hutchinson_samples=100,
        beta=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent = latent
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.hutchinson_samples = hutchinson_samples
        self.beta = torch.Tensor([beta])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_scheduler == "cosine":
            # cosine learning rate annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.lr_min,
                verbose=True,
            )
        elif self.lr_scheduler == "step":
            # An scheduler is optional, but can help in flows to get the last bpd improvement
            scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        else:
            scheduler = None
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # get the surrogate loss, latent representation, and reconstructed tensor
        surrogate_loss, v_hat, x_hat = volume_change_surrogate_transformer(
            x,
            self.encoder,
            self.decoder,
            hutchinson_samples=self.hutchinson_samples,
        )
        # compute reconstruction loss
        loss_reconstruction = torch.nn.functional.mse_loss(x_hat, x)

        # get negative log likelihoood
        embed_dim = self.decoder.embed_dim
        v_hat = v_hat.view(-1, embed_dim)
        loss_nll = -self.latent.log_prob(v_hat).mean() - surrogate_loss

        loss = self.beta * loss_reconstruction + loss_nll

        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def sample(self, num_samples=1, **params):
        """
        Sample a batch of images from the flow.
        """
        # sample latent space and reshape to (batches, 1, embed_dim)
        v, _ = self.latent.sample(num_samples, **params)
        v = v.reshape(num_samples, 1, -1)
        return self.decoder(v)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        self.beta = self.beta.to(x.device)

        # get the surrogate loss, latent representation, and reconstructed tensor
        surrogate_loss, v_hat, x_hat = volume_change_surrogate_transformer(
            x,
            self.encoder,
            self.decoder,
            hutchinson_samples=self.hutchinson_samples,
        )
        # compute reconstruction loss
        loss_reconstruction = torch.nn.functional.mse_loss(x_hat, x).to(x.device)

        # get negative log likelihoood
        loss_nll = (-self.latent.log_prob(v_hat).mean() - surrogate_loss).to(x.device)

        loss = self.beta * loss_reconstruction + loss_nll
        # Print the loss to the console
        if batch_idx % 100 == 0:
            print(f"val_loss: {loss}")
        self.log("val_loss", loss)
        return loss


class plCausalModel(plFlowModel):
    def training_step(self, batch, batch_idx):
        samples = batch[0]
        meta_labels = batch[1]
        distr_idx = meta_labels[:, -1]
        hard_interventions = None
        targets = batch[2]

        surrogate_loss, v_hat, x_hat = volume_change_surrogate(
            samples, self.model.encoder, self.model.decoder, hutchinson_samples=1000
        )

        # print("Inside training step: ", samples.shape, meta_labels.shape, targets.shape)
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self.model.forward_kld(
            samples,
            intervention_targets=targets,
            distr_idx=distr_idx,
            hard_interventions=hard_interventions,
        )
        self.log("train_kld", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # first element of batch is the image tensor
        # second element of batch is the label tensor
        #     "width",
        #     "color",
        #     "fracture_thickness",
        #     "fracture_num_fractures",
        #     "label",
        #     "distr_indicators",
        # third element of batch is the intervention target
        samples = batch[0]
        meta_labels = batch[1]
        distr_idx = meta_labels[:, -1]
        hard_interventions = None
        targets = batch[2]
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self.model.forward_kld(
            samples,
            intervention_targets=targets,
            distr_idx=distr_idx,
            hard_interventions=hard_interventions,
        )

        self.log("val_kld", loss)
        return loss
