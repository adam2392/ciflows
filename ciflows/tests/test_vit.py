import pytest
import torch

from ciflows.vit import (PatchEmbedding, TransformerEncoderLayer,
                         VisionTransformerDecoder, VisionTransformerEncoder)


def test_patch_embedding_output_shape():
    patch_embedding = PatchEmbedding(
        img_size=224, patch_size=16, in_channels=3, embed_dim=768
    )

    # Test that output shape is correct for valid input
    B, C, H, W = 1, 3, 224, 224  # Batch size 1, 3 channels, 224x224 image
    input_tensor = torch.randn(B, C, H, W)

    output = patch_embedding(input_tensor)

    expected_num_patches = (224 // 16) ** 2  # 196 patches
    expected_embed_dim = 768

    assert output.shape == (
        B,
        expected_num_patches,
        expected_embed_dim,
    ), f"Expected output shape {(B, expected_num_patches, expected_embed_dim)}, but got {output.shape}"

    patch_embedding = PatchEmbedding(
        img_size=28, patch_size=4, in_channels=3, embed_dim=768
    )
    # Test that output shape is correct for valid input
    B, C, H, W = 1, 3, 28, 28  # Batch size 1, 3 channels, 224x224 image
    input_tensor = torch.randn(B, C, H, W)

    output = patch_embedding(input_tensor)

    expected_num_patches = (28 // 4) ** 2  # 196 patches
    expected_embed_dim = 768

    assert output.shape == (
        B,
        expected_num_patches,
        expected_embed_dim,
    ), f"Expected output shape {(B, expected_num_patches, expected_embed_dim)}, but got {output.shape}"


def test_transformer_encoder_layer_output_shape():
    transformer_encoder_layer = TransformerEncoderLayer(
        embed_dim=768, n_heads=8, hidden_dim=1024, dropout=0.1
    )

    # Test that the output shape is correct
    seq_length, batch_size, embed_dim = (
        5,
        2,
        768,
    )  # Example sequence length, batch size, and embedding dim
    input_tensor = torch.randn(
        batch_size, seq_length, embed_dim
    )  # Input shape (S, B, E)

    output = transformer_encoder_layer(input_tensor)

    # Assert output shape matches input shape
    assert (
        output.shape == input_tensor.shape
    ), f"Expected output shape {input_tensor.shape}, but got {output.shape}"
    print(output.shape)
    assert output.requires_grad, "Output should require grad if input requires grad"


def test_vision_transformer_encoder_output_shape():
    vision_transformer_encoder = VisionTransformerEncoder(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        n_heads=8,
        hidden_dim=1024,
        n_layers=6,
        dropout=0.1,
    )

    # Test that the output shape is correct
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)  # Input shape (B, C, H, W)

    output = vision_transformer_encoder(input_tensor)

    num_patches = (224 // 16) ** 2  # 196 patches
    expected_shape = (batch_size, num_patches, 768)  # +1 for CLS token

    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"


def test_vit_decoder():
    # Example Usage
    encoder_output = torch.randn(1, 49, 768)  # (B, n_patches, embed_dim)
    decoder = VisionTransformerDecoder(
        img_size=28,
        patch_size=4,
        in_channels=3,
        embed_dim=768,
        n_heads=2,
        hidden_dim=1024,
        n_layers=2,
        dropout=0.1,
    )
    reconstructed_img = decoder(encoder_output)
    print(
        reconstructed_img.shape
    )  # Output will be [batch_size, channels, height, width]
    assert reconstructed_img.shape == (
        1,
        49,
        3,
        28,
        28,
    ), f"Expected output shape {(1, 49, 3, 28, 28)}, but got {reconstructed_img.shape}"


def test_vit_decoder_from_latent():
    # Example Usage
    encoder_output = torch.randn(16, 1, 768)  # (B, n_patches, embed_dim)
    decoder = VisionTransformerDecoder(
        img_size=28,
        patch_size=4,
        in_channels=3,
        embed_dim=768,
        n_heads=2,
        hidden_dim=1024,
        n_layers=2,
        dropout=0.1,
    )
    reconstructed_img = decoder(encoder_output)
    print(
        reconstructed_img.shape
    )  # Output will be [batch_size, channels, height, width]
    assert reconstructed_img.shape == (
        16,
        1,
        3,
        28,
        28,
    ), f"Expected output shape {(16, 1, 3, 28, 28)}, but got {reconstructed_img.shape}"
