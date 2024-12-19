import logging
from pathlib import Path

import lightning as pl
import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import transforms

from ciflows.datasets.lightning import DatasetName, MultiDistrDataModule
from ciflows.distributions.pgm import LinearGaussianDag
from ciflows.flows import TwoStageTraining, plCausalInjFlowModel
from ciflows.flows.glow import GlowBlock, InjectiveGlowBlock, ReshapeFlow, Squeeze
from ciflows.flows.model import CausalNormalizingFlow


def get_inj_model(input_shape):
    use_lu = True
    gamma = 1e-3
    activation = "linear"
    dropout_probability = 0.2

    net_actnorm = False
    # n_hidden_list = [32, 64, 128, 256, 256, 256]
    n_hidden = 128
    n_glow_blocks = 2
    n_mixing_layers = 5
    n_injective_layers = 10
    n_layers = n_mixing_layers + n_injective_layers

    # hidden layers for the AutoregressiveRationalQuadraticSpline
    net_hidden_layers = 2
    net_hidden_dim = 32

    n_channels = input_shape[0]
    img_size = input_shape[1]

    n_chs = n_channels
    flows = []

    debug = False

    n_chs = int(n_channels * 4**n_mixing_layers * (1 / 2) ** n_injective_layers)
    latent_size = int(img_size / (2**n_mixing_layers))
    init_n_chs = n_chs
    init_latent_size = latent_size
    print("Starting at latent representation: ", n_chs, "with latent size: ", latent_size)
    q0 = nf.distributions.DiagGaussian((n_chs, latent_size, latent_size), trainable=False)

    split_mode = "channel"

    for i in range(n_injective_layers):
        # n_hidden = n_hidden_list[-i]
        if i <= 1:
            split_mode = "checkerboard"
        else:
            split_mode = "channel"

        if i % 1 == 0:
            for j in range(n_glow_blocks):
                flows += [
                    GlowBlock(
                        channels=n_chs,
                        hidden_channels=n_hidden,
                        use_lu=use_lu,
                        scale=True,
                        split_mode=split_mode,
                        net_actnorm=net_actnorm,
                        dropout_probability=dropout_probability,
                    )
                ]
        else:
            flows += [
                ReshapeFlow(
                    shape_in=(n_chs, latent_size, latent_size),
                    shape_out=(n_chs * latent_size * latent_size,),
                )
            ]
            flows += [
                nf.flows.AutoregressiveRationalQuadraticSpline(
                    num_input_channels=n_chs * latent_size * latent_size,
                    num_blocks=net_hidden_layers,
                    num_hidden_channels=net_hidden_dim,
                    permute_mask=True,
                )
            ]
            flows += [
                ReshapeFlow(
                    shape_in=(n_chs * latent_size * latent_size,),
                    shape_out=(n_chs, latent_size, latent_size),
                )
            ]

        # input to inj flow is what is at the X -> V layer
        flows += [
            InjectiveGlowBlock(
                channels=n_chs,
                hidden_channels=n_hidden,
                activation=activation,
                scale=True,
                gamma=gamma,
                debug=debug,
                split_mode=split_mode,
                net_actnorm=net_actnorm,
            )
        ]
        n_chs = n_chs * 2
        latent_size = latent_size
        if debug:
            print(f"On layer {n_layers - i}, n_chs = {n_chs//2} -> {n_chs}")

    # split_mode = "channel_inv"
    for i in range(n_mixing_layers):
        # if i > 0:# n_mixing_layers - 1:
        for j in range(n_glow_blocks):
            flows += [
                GlowBlock(
                    channels=n_chs,
                    hidden_channels=n_hidden,
                    use_lu=use_lu,
                    scale=False,
                    split_mode=split_mode,
                    dropout_probability=dropout_probability,
                )
            ]
        # else:
        #     flows += [
        #         ReshapeFlow(
        #             shape_in=(n_chs, latent_size, latent_size),
        #             shape_out=(n_chs * latent_size * latent_size,),
        #         )
        #     ]
        #     flows += [
        #         nf.flows.AutoregressiveRationalQuadraticSpline(
        #             num_input_channels=n_chs * latent_size * latent_size,
        #             num_blocks=net_hidden_layers,
        #             num_hidden_channels=net_hidden_dim,
        #             permute_mask=True,
        #         )
        #     ]
        #     flows += [
        #         ReshapeFlow(
        #             shape_in=(n_chs * latent_size * latent_size,),
        #             shape_out=(n_chs, latent_size, latent_size),
        #         )
        #     ]

        flows += [Squeeze()]
        n_chs = n_chs // 4
        latent_size *= 2
        if debug:
            print(f"On layer {n_mixing_layers - i}, n_chs = {n_chs}")

    model = nf.NormalizingFlow(q0=q0, flows=flows)
    model.output_n_chs = init_n_chs
    model.output_latent_size = init_latent_size
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    return model


def get_bij_model(
    n_chs,
    latent_size,
    adj_mat,
    cluster_sizes,
    intervention_targets,
    confounded_variables,
):
    use_lu = True
    net_actnorm = False
    n_hidden = 128
    n_glow_blocks = 10

    flows = []

    debug = False

    print("Starting at latent representation: ", n_chs, latent_size, latent_size)
    print("Got Intervention targets for q0: ", intervention_targets)
    # q0 = nf.distributions.DiagGaussian(
    #     (n_chs, latent_size, latent_size), trainable=False
    # )
    q0 = nf.distributions.DiagGaussian((n_chs * latent_size * latent_size,), trainable=False)

    # q0 = ClusteredLinearGaussianDistribution(
    #     adjacency_matrix=adj_mat,
    #     cluster_sizes=cluster_sizes,
    #     intervention_targets_per_distr=intervention_targets,
    #     hard_interventions_per_distr=None,
    #     confounded_variables=confounded_variables,
    # )
    node_dimensions = {
        0: 16,
        1: 16,
        2: 16,
    }
    edge_list = [(1, 2)]
    noise_means = {
        0: torch.zeros(16),
        1: torch.zeros(16),
        2: torch.zeros(16),
    }
    noise_variances = {
        0: torch.ones(16),
        1: torch.ones(16),
        2: torch.ones(16),
    }
    intervened_node_means = [{2: torch.ones(16) + 2}, {2: torch.ones(16) + 4}]
    intervened_node_vars = [{2: torch.ones(16)}, {2: torch.ones(16) + 2}]

    confounded_list = []
    # independent noise with causal prior
    q0 = LinearGaussianDag(
        node_dimensions=node_dimensions,
        edge_list=edge_list,
        noise_means=noise_means,
        noise_variances=noise_variances,
        confounded_list=confounded_list,
        intervened_node_means=intervened_node_means,
        intervened_node_vars=intervened_node_vars,
    )

    split_mode = "checkerboard"

    net_hidden_layers = 2
    net_hidden_dim = 32

    # flows += [
    #     ReshapeFlow(
    #         shape_in=(n_chs, latent_size, latent_size),
    #         shape_out=(n_chs * latent_size * latent_size,),
    #     )
    # ]
    # n_chs = n_chs * latent_size * latent_size

    # using glow blocks
    # n_chs = int(n_chs * 4**n_glow_blocks)
    for i in range(n_glow_blocks):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        # n_chs *= 4
        # param_map = nf.nets.MLP([n_chs, net_hidden_dim, n_chs*2], init_zeros=True)
        # # # Add flow layer
        # flows.append(nf.flows.AffineCouplingBlock(param_map, split_mode='channel'))
        # flows.append(nf.flows.Permute(n_chs, mode='swap'))

        # Autoregressive Neural Spline flow
        # Swap dimensions
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                num_input_channels=n_chs * latent_size * latent_size,
                num_blocks=net_hidden_layers,
                num_hidden_channels=net_hidden_dim,
                permute_mask=True,
            )
        ]
        # if i  1:
        #     split_mode = "checkerboard"
        # else:
        #     split_mode = "channel"
        # flows += [
        #     GlowBlock(
        #         channels=n_chs,
        #         hidden_channels=n_hidden,
        #         use_lu=use_lu,
        #         scale=True,
        #         split_mode=split_mode,
        #         net_actnorm=net_actnorm,
        #         dropout_probability=0.2,
        #     )
        # ]
        # flows += [Squeeze()]
        # n_chs = n_chs // 4
        # print(n_chs)
        if debug:
            print(f"On layer {n_glow_blocks - i}, n_chs = {n_chs//2} -> {n_chs}")

    # maps x to (n_chs * latent_size * latent_size), while v is mapped to (n_chs, latent_size, latent_size)
    flows += [
        ReshapeFlow(
            shape_in=(n_chs * latent_size * latent_size,),
            shape_out=(n_chs, latent_size, latent_size),
        )
    ]
    # model = nf.NormalizingFlow(q0=q0, flows=flows)
    model = CausalNormalizingFlow(q0=q0, flows=flows)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    return model


def initialize_flow(model):
    """
    Initialize a full normalizing flow model
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            # Layer-dependent initialization
            if "coupling" in name:
                nn.init.normal_(param, mean=0.0, std=1.0)
            elif "batch" in name:
                continue
            else:
                nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 1e-5)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # log_dir = args.log_dir
    # log_dir = '/home/adam2392/projects/logs/'
    seed = 1234

    # set seed
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)

    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    print(f"Using device: {device}")
    print(f"Using accelerator: {accelerator}")

    debug = False
    fast_dev = False
    image_size = 128
    input_shape = (3, image_size, image_size)
    max_epochs = 2000
    batch_size = 256
    devices = 1
    strategy = "auto"  # or ddp if distributed
    num_workers = 6
    gradient_clip_val = 1.0
    check_val_every_n_epoch = 1
    check_samples_every_n_epoch = 5
    monitor = "val_loss"

    n_steps_mse = 15
    mse_chkpoint_name = f"mse_chkpoint_{n_steps_mse}"

    lr = 3e-4
    lr_min = 1e-8
    lr_scheduler = "cosine"

    torch.set_float32_matmul_precision("high")
    if debug:
        accelerator = "cpu"
        fast_dev = True
        max_epochs = 5
        n_steps_mse = 2
        batch_size = 16
        check_samples_every_n_epoch = 1
        num_workers = 2

    # whether or not to shuffle dataset
    shuffle = True

    graph_type = "chain"
    # gender, age, haircolor
    adj_mat = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    confounded_variables = None
    cluster_sizes = [16, 16, 16]

    # output filename for the results
    if debug:
        root = Path("/Users/adam2392/pytorch_data/")
    else:
        root = Path("/home/adam2392/projects/data/")

    # v2 = trainable q0
    # v3 = also make 512 latent dim, and fix initialization of coupling to 1.0 standard deviation
    # convnet restart = v2, whcih was good
    model_name = "16dimlatent_10layerneuralspline_twostage_batch256_gradclip1_causalcelebadim128_nstepsmse10_v2"
    checkpoint_dir = root / "CausalCelebA" / "causalinjflow" / model_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    train_from_checkpoint = False

    # if not debug:
    #     model = torch.compile(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=5,
        monitor=monitor,
        every_n_epochs=check_val_every_n_epoch,
    )

    # logger = TensorBoardLogger(
    #     "check_injflow_mnist_logs",
    #     name="check_injflow_mnist",
    #     version="01",
    #     log_graph=True,
    # )
    # logger = logging.getLogger()
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    print()
    print(f"Model name: {model_name}")
    print()
    # demo to load dataloader. please make sure transform is None. d
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            # discretize,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    data_module = MultiDistrDataModule(
        root=root,
        graph_type=graph_type,
        batch_size=batch_size,
        stratify_distrs=True,
        num_workers=num_workers,
        transform=transform,
        dataset_name=DatasetName.CAUSAL_CELEBA,
        fast_dev_run=fast_dev,
    )
    data_module.setup()

    intervention_targets_per_distr = []
    print()
    print("Intervention target summary: ")
    print(data_module.dataset.intervention_targets.shape)
    for distr_idx in data_module.dataset.distribution_idx.unique():
        idx = np.argwhere(data_module.dataset.distribution_idx == distr_idx)[0][0]
        intervention_targets_per_distr.append(data_module.dataset.intervention_targets[idx])
    print(idx)
    print(intervention_targets_per_distr)

    intervention_targets_per_distr = np.array(intervention_targets_per_distr)
    unique_rows = np.unique(data_module.dataset.intervention_targets, axis=0)
    print("Unique intervention targets: ", unique_rows)
    print()

    if train_from_checkpoint:
        epoch = 499
        step = 27000
        model_fname = checkpoint_dir / f"epoch={epoch}-step={step}.ckpt"
        model = plCausalInjFlowModel.load_from_checkpoint(model_fname)

        model_name = "16dimlatent_10layerneuralspline_twostage_batch256_gradclip1_causalmnist_nottrainableq0_nstepsmse10_v1"

        # model.current_epoch = epoch
        max_epochs = model.current_epoch + 1000
        fast_dev = False
        debug = False
    else:
        model_fname = None
        # define the model
        inj_model = get_inj_model(input_shape=input_shape)
        samples = inj_model.q0.sample(2)
        # _, n_chs, latent_size, _ = samples.shape
        n_chs = inj_model.output_n_chs
        latent_size = inj_model.output_latent_size
        assert samples.shape == (
            2,
            n_chs,
            latent_size,
            latent_size,
        ), f"Expected shape: {(batch_size, n_chs, latent_size, latent_size)}, got {samples.shape}"
        print("Output shape of injective flow model: ", samples.shape)
        initialize_flow(inj_model)

        # bij_model = None
        bij_model = get_bij_model(
            n_chs=n_chs,
            latent_size=latent_size,
            adj_mat=adj_mat,
            cluster_sizes=cluster_sizes,
            intervention_targets=intervention_targets_per_distr,
            confounded_variables=confounded_variables,
        )
        initialize_flow(bij_model)

        example_input_array = [
            torch.randn(2, n_chs * latent_size * latent_size),
            torch.randn(2, 1),
        ]
        model = plCausalInjFlowModel(
            inj_model=inj_model,
            bij_model=bij_model,
            lr=lr,
            lr_min=lr_min,
            lr_scheduler=lr_scheduler,
            n_steps_mse=n_steps_mse,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=mse_chkpoint_name,
            debug=debug,
            check_val_every_n_epoch=check_val_every_n_epoch,
            check_samples_every_n_epoch=check_samples_every_n_epoch,
            gradient_clip_val=gradient_clip_val,
            example_input_array=example_input_array,
        )

        # test the forward and inverse pass
        test_tensor = torch.randn(2, *input_shape)
        test_sample = model.bij_model.inverse(model.inverse(test_tensor))
        print(test_sample.shape)

        test_latent_tensor = torch.randn(2, n_chs * latent_size * latent_size)
        print(test_latent_tensor.shape)
        test_sample = model.inj_model.forward(model.bij_model.forward(test_latent_tensor))
        print(test_sample.shape)
        print("Test passed!")

    # Define the trainer
    trainer = pl.Trainer(
        logger=False,
        max_epochs=max_epochs,
        devices=devices,
        strategy=strategy,
        callbacks=[checkpoint_callback, TwoStageTraining()],
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
        gradient_clip_val=gradient_clip_val,
        # precision="bf16",
        fast_dev_run=fast_dev,
        # log_every_n_steps=1,
        # max_epochs=1,
        # limit_train_batches=1,
        # limit_val_batches=1,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        # ckpt_path=model_fname
    )

    # save the final model
    print(f"Saving model to {checkpoint_dir / '{model_name}_final.pt'}")
    torch.save(model, checkpoint_dir / f"{model_name}_final.pt")
