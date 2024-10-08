import lightning as pl
import torch
import torch.optim as optim
from lightning.pytorch.callbacks import Callback

from .glow import InjectiveGlowBlock
from .loss import volume_change_surrogate
from .model import InjectiveFlow


class TwoStageTraining(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if (
            getattr(pl_module, "n_steps_mse", None) is not None
            and trainer.current_epoch > pl_module.n_steps_mse
        ):
            trainer.optimizers = [pl_module.optimizer_nll]
            # trainer.lr_schedulers = trainer.configure_schedulers([pl_module.])
            # trainer.optimizer_frequencies = [] # or optimizers frequencies if you have any
        else:
            trainer.optimizers = [pl_module.optimizer_mse]
            # trainer.lr_schedulers = trainer.configure_schedulers([pl_module.optimizer_mse])
            # trainer.optimizer_frequencies = [] # or optimizers frequencies if you have any


class plFlowModel(pl.LightningModule):
    def __init__(
        self,
        model: InjectiveFlow,
        lr: float = 1e-3,
        lr_min: float = 1e-8,
        lr_scheduler=None,
        n_steps_mse=None,
    ):
        """Injective flow model lightning module.

        Parameters
        ----------
        model : FlowModel
            An (injective) flow model.
        lr : float, optional
            Learning rate for SGD, by default 1e-3.
        lr_min : float, optional
            Min learning rate, by default 1e-8.
        lr_scheduler : str, optional
            A learning rate scheduler, by default None. Can be 'cosine', or 'step'.
        n_steps_mse : int

        """
        super().__init__()

        # ensure that the model is saved and can be loaded later as a checkpoint
        self.save_hyperparameters()

        # XXX: This should change depending on the dataset
        self.example_input_array = [torch.randn(2, 1, 28, 28), torch.randn(2, 1)]

        self.model = model

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.n_steps_mse = n_steps_mse
        self.step_counter = 0

    def get_injective_and_other_params(self):
        injective_params = []
        other_params = []

        for flow in self.model.flows:
            # Check if the flow is an injective Glow block
            if isinstance(flow, InjectiveGlowBlock):
                injective_params += list(flow.parameters())
            else:
                other_params += list(flow.parameters())

        return injective_params, other_params

    @torch.no_grad()
    def sample(self, num_samples=1, **params):
        """
        Sample a batch of images from the flow.
        """
        return self.model.sample(num_samples=num_samples, **params)

    def forward(self, x, target=None) -> torch.Any:
        """Foward pass.

        Note: This is opposite of the normalizing flow API convention.
        """
        return self.model.inverse(x)

    def inverse(self, v, target=None) -> torch.Any:
        """Inverse pass.

        Note: This is opposite of the normalizing flow API convention.
        """
        return self.model.forward(v)

    def configure_optimizers(self):
        mse_params, nll_params = self.get_injective_and_other_params()
        optimizer_mse = optim.Adam(mse_params, lr=self.lr)
        optimizer_nll = optim.Adam(nll_params, lr=self.lr)

        self.optimizer_mse = optimizer_mse
        self.optimizer_nll = optimizer_nll
        scheduler_list = []
        optimizer_list = [optimizer_mse, optimizer_nll]
        for optimizer in optimizer_list:
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
            scheduler_list.append(scheduler)
        # return optimizer_list, scheduler_list
        return [optimizer_nll], [scheduler]

    def training_step(self, batch, batch_idx):
        x, _ = batch

        if self.n_steps_mse is not None and self.step_counter < self.n_steps_mse:
            v_latent = self.model.inverse(x)
            x_reconstructed = self.model.forward(v_latent)
            loss = torch.nn.functional.mse_loss(x_reconstructed, x)
        else:
            loss = self.model.forward_kld(x)

        # logging the loss
        self.log("train_loss", loss)
        if batch_idx % 100 == 0:
            print(f"train_loss: {loss}")
        self.step_counter += 1
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        if self.n_steps_mse is not None and self.step_counter < self.n_steps_mse:
            v_latent = self.model.inverse(x)
            x_reconstructed = self.model.forward(v_latent)
            loss = torch.nn.functional.mse_loss(x_reconstructed, x)
        else:
            loss = self.model.forward_kld(x)

        self.log("Nsteps_mse", self.n_steps_mse)
        self.log("step_counter", self.step_counter)
        self.log("val_loss", loss)

        # Print the loss to the console
        if batch_idx % 100 == 0:
            print(
                f"Nsteps_mse {self.n_steps_mse}, step_counter: {self.step_counter}, val_loss: {loss}"
            )
        return loss


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
        surrogate_loss, v_hat, x_hat = volume_change_surrogate(
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
        # sample latent space
        v = self.latent.sample((num_samples,), **params)
        return self.decoder(v)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        self.beta = self.beta.to(x.device)

        # get the surrogate loss, latent representation, and reconstructed tensor
        surrogate_loss, v_hat, x_hat = volume_change_surrogate(
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
