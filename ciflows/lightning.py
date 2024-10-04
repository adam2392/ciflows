import lightning as pl
import torch
import torch.optim as optim

from .loss import volume_change_surrogate
from .model import InjectiveFlow


class plFlowModel(pl.LightningModule):
    def __init__(
        self,
        model: InjectiveFlow,
        lr: float = 1e-3,
        lr_min: float = 1e-8,
        lr_scheduler=None,
    ):
        """Causal flow model lightning module.

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
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        # XXX: This should change depending on the dataset
        self.example_input_array = [torch.randn(2, 1, 28, 28), torch.randn(2, 1)]

        self.model = model

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min

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
        loss = self.model.forward_kld(x)

        # logging the loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.model.forward_kld(x)

        self.log("val_loss", loss)
        return loss


# TODO: finish implementation
class plApproximateFlowModel(pl.LightningModule):
    """Approximate flow model lightning module.

    Here, we define an approximate flow model for training.
    """

    def __init__(self, model, lr=1e-3, lr_min=1e-8, lr_scheduler=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min


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
