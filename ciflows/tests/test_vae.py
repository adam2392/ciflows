import pytest
import torch
from ciflows.vae import Encoder, Decoder


@pytest.mark.parametrize("input_channels", [1, 3])
@pytest.mark.parametrize(
    "img_size",
    [
        28,
        # 64, 128
    ],
)
def test_encoder_decoder_output_shape(input_channels, img_size):
    # input_channels = 3  # e.g., for RGB images
    # img_size = 64  # Input image size (64x64 in this example)
    embed_dim = 12  # Desired embedding dimension

    encoder = Encoder(
        input_channels,
        img_size,
        embed_dim,
        hidden_dim=1024,
        dropout=0.1,
        n_bottleneck_layers=5,
        n_compression_layers=2,
        debug=True,
    )

    # Test with a random image tensor (e.g., batch size of 8, 3 channels, 64x64 image)
    test_input = torch.randn(8, input_channels, img_size, img_size)

    encoding = encoder(test_input)

    assert encoding.shape == (
        8,
        embed_dim,
    ), f"Expected output shape (8, {embed_dim}), but got {encoding.shape}"

    print("Now doing decoder:")
    decoder = Decoder(
        embed_dim,
        img_size,
        input_channels,
        n_upsample_layers=2,
        n_bottleneck_layers=5,
        hidden_dim=1024,
        dropout=0.1,
        debug=True,
    )
    decoded_img = decoder(encoding)
    assert decoded_img.shape == (
        8,
        input_channels,
        img_size,
        img_size,
    ), f"Expected output shape (8, {input_channels}, {img_size}, {img_size}), but got {decoded_img.shape}"
