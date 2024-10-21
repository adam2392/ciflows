import os

import lightning as pl
import normflows as nf
import torch
import torch.optim as optim
from lightning.pytorch.callbacks import Callback

from ..loss import volume_change_surrogate, volume_change_surrogate_transformer
from .glow import Injective1x1Conv, InjectiveGlowBlock
from .model import InjectiveFlow


class TwoStageTraining(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if (
            getattr(pl_module, "n_steps_mse", None) is not None
            and trainer.current_epoch > pl_module.n_steps_mse
        ):
            if trainer.current_epoch == pl_module.n_steps_mse + 1:
                print()
                print("Training with NLL loss")
            trainer.optimizers = [pl_module.optimizer_nll]
            # trainer.lr_schedulers = trainer.configure_schedulers([pl_module.])
            # trainer.optimizer_frequencies = [] # or optimizers frequencies if you have any

            # Save a checkpoint after the transition to NLL
            checkpoint_path = os.path.join(
                pl_module.checkpoint_dir, f"{pl_module.checkpoint_name}.ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        else:
            if trainer.current_epoch == 0:
                print()
                print("Training with MSE loss")
            trainer.optimizers = [pl_module.optimizer_mse]
            # trainer.lr_schedulers = trainer.configure_schedulers([pl_module.optimizer_mse])
            # trainer.optimizer_frequencies = [] # or optimizers frequencies if you have any


class plInjFlowModel(pl.LightningModule):
    def __init__(
        self,
        inj_model: InjectiveFlow,
        bij_model: nf.NormalizingFlow,
        lr: float = 1e-3,
        lr_min: float = 1e-8,
        lr_scheduler=None,
        n_steps_mse=None,
        checkpoint_dir=None,
        checkpoint_name=None,
        debug=False,
        check_val_every_n_epoch=1,
    ):
        """Injective flow model lightning module.

        Parameters
        ----------
        inj_model : FlowModel
            An (injective) flow model.
        bij_model : NormalizingFlow
            A bijection model.
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
        self.example_input_array = [torch.randn(2, 1, 8, 8), torch.randn(2, 1)]

        self.inj_model = inj_model
        self.bij_model = bij_model

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.n_steps_mse = n_steps_mse
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.debug = debug
        self.check_val_every_n_epoch = check_val_every_n_epoch

    # def get_injective_and_other_params(self):
    #     # injective_params = []
    #     # other_params = []

    #     # for flow in self.model.flows:
    #     #     # Check if the flow is an injective Glow block
    #     #     # print(flow._get_name())

    #     #     if isinstance(flow, InjectiveGlowBlock):
    #     #         for block in flow.flows:
    #     #             if isinstance(block, Injective1x1Conv):
    #     #                 injective_params += list(block.parameters())
    #     #             else:
    #     #                 other_params += list(block.parameters())
    #     #     else:
    #     #         other_params += list(flow.parameters())
    #     # Get all trainable parameters in the model
    #     all_params = list(self.model.parameters())

    #     # Collect injective parameters
    #     injective_params = []
    #     for flow in self.model.flows:
    #         if isinstance(flow, InjectiveGlowBlock):
    #             injective_params += list(flow.parameters())
    #             # for block in flow.flows:
    #             #     if isinstance(block, Injective1x1Conv):
    #             #         injective_params += list(block.parameters())

    #     # Convert injective_params to a set for set operations
    #     injective_params_set = set(injective_params)

    #     # Use set difference to get non-injective (other) parameters
    #     other_params = [p for p in all_params if p not in injective_params_set]

    #     return injective_params, other_params

    @torch.no_grad()
    def sample(self, num_samples=1, **params):
        """
        Sample a batch of images from the flow.
        """
        samples, ldj = self.bij_model.sample(num_samples=num_samples, **params)
        return self.inj_model.forward(samples), ldj

    def forward(self, x, target=None):
        """Foward pass.

        Note: This is opposite of the normalizing flow API convention.
        """
        # second pass through bijection layer
        x = self.bij_model.forward(x)
        # first pass through injective layer
        x = self.inj_model.forward(x)
        return x

    def inverse(self, v, target=None):
        """Inverse pass.

        Note: This is opposite of the normalizing flow API convention.
        """
        v = self.inj_model.inverse(v)
        v = self.bij_model.inverse(v)
        return v

    def configure_optimizers(self):
        # mse_params, nll_params = self.get_injective_and_other_params()
        mse_params = list(self.inj_model.parameters())
        nll_params = list(self.bij_model.parameters())
        # Check the number of parameters in each optimizer
        num_mse_params = sum(p.numel() for p in mse_params)
        num_nll_params = sum(p.numel() for p in nll_params)

        print(f"Number of parameters in optimizer_mse: {num_mse_params}")
        print(f"Number of parameters in optimizer_nll: {num_nll_params}")
        print(f"Total number of parameters: {num_mse_params + num_nll_params}")
        # trainable_params = sum(
        #     p.numel() for p in self.inj_model.parameters() if p.requires_grad
        # )
        # print(f"Total number of trainable parameters: {trainable_params}")
        # assert trainable_params == num_mse_params + num_nll_params

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

        if self.n_steps_mse is not None and self.current_epoch < self.n_steps_mse:
            v_latent = self.inj_model.inverse(x)
            x_reconstructed = self.inj_model.forward(v_latent)

            # reconstruct the latents
            v_latent_recon = self.inj_model.inverse(x_reconstructed)

            # check if any nans
            if torch.isnan(x_reconstructed).any():
                print(
                    "x_reconstructed has nans", x_reconstructed, x_reconstructed.shape
                )
            if torch.isnan(v_latent_recon).any():
                print("v_latent_recon has nans", v_latent_recon, v_latent_recon.shape)

            loss = torch.nn.functional.mse_loss(
                x_reconstructed, x
            ) + torch.nn.functional.mse_loss(v_latent_recon, v_latent)
        else:
            inj_v = self.inj_model.inverse(x)
            loss = self.bij_model.forward_kld(inj_v)

        # logging the loss
        self.log("train_loss", loss)
        if self.current_epoch % self.check_val_every_n_epoch == 0 and batch_idx == 0 or self.debug:
            print()
            print(f"train_loss: {loss} | epoch_counter: {self.current_epoch}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        if self.n_steps_mse is not None and self.current_epoch < self.n_steps_mse:
            v_latent = self.inj_model.inverse(x)
            x_reconstructed = self.inj_model.forward(v_latent)
            # reconstruct the latents
            v_latent_recon = self.inj_model.inverse(x_reconstructed)

            loss = torch.nn.functional.mse_loss(
                x_reconstructed, x
            ) + torch.nn.functional.mse_loss(v_latent_recon, v_latent)
        else:
            inj_v = self.inj_model.inverse(x)
            loss = self.bij_model.forward_kld(inj_v)

        self.log("Nsteps_mse", self.n_steps_mse)
        self.log("val_loss", loss)

        # Print the loss to the console
        if self.current_epoch % self.check_val_every_n_epoch == 0 and batch_idx == 0 or self.debug:
            print()
            print(
                f"Nsteps_mse {self.n_steps_mse}, epoch_counter: {self.current_epoch}, val_loss: {loss}"
            )
        return loss
