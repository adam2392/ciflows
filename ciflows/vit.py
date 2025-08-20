# Implementation of Vision Transformer Encoder and Decoder.
#
# This provides an API to allow matching dimensionality with
# the latent space distribution.
import torch
import torch.nn as nn
from torch import Tensor


def get_img_patches(x: Tensor, patch_size: int):
    B, C, H, W = x.shape
    img_size = H
    num_patches = (img_size // patch_size) ** 2

    # split the tensor into patches as a 6D tensor with (B, C, n_patches_in_height, n_patches_in_width, patch_size, patch_size)
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # B, C, n_patches, patch_dim * patch_dim
    x = x.contiguous().view(B, C, -1, patch_size * patch_size)

    # B, n_patches, C * patch_dim * patch_dim
    x = x.permute(0, 2, 1, 3).reshape(B, num_patches, -1)
    return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size=28,
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # number of patches = Height/Width / size of the patch in that dimension * itself
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass to embed the input image.

        Parameters
        ----------
        x : Tensor of shape (B, C, H, W)
            Input images.

        Returns
        -------
        Tensor of shape (B, n_patches, embed_dim)
            Embedded patches.
        """
        B, C, H, W = x.shape
        assert (
            H == W == self.img_size
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        # split the tensor into patches as a 6D tensor with (B, C, n_patches_in_height, n_patches_in_width, patch_size, patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        # B, C, n_patches, patch_dim * patch_dim
        x = x.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
        # B, n_patches, C * patch_dim * patch_dim
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, -1)
        return self.proj(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffm = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # self-attention
        x = self.norm1(x)
        x = x + self.self_attn(x, x, x)[0]

        # feed-forward network
        x = x + self.ffm(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embed_dim,
        n_heads,
        hidden_dim,
        n_layers,
        dropout=0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(img_size, in_channels, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, embed_dim))

        # how many layers of transformer encoder to stack
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, n_heads, hidden_dim, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        """Forward pass to encode the input image.

        Parameters
        ----------
        x : torch.Tensor of shape (B, C, H, W)
            The input image tensor.

        Returns
        -------
        torch.Tensor of shape (B, n_patches, embed_dim)
            The embeddings.
        """
        # patch embed and positional embedding
        # (B, C, H, W) -> (B, n_patches, embed_dim)
        patch_embedding = self.patch_embed(x)
        x = patch_embedding + self.pos_embed

        # pass through the transformer encoder layers
        global_residual = x
        for layer in self.encoder_layers:
            x = layer(x)

        # residual connection
        return x + global_residual


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, dropout=0.0):
        super().__init__()
        # self-attention, cross-attention, feed-forward
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ffm = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, encoder_output):
        # self-attention
        x = self.norm1(x)
        x = x + self.self_attn(x, x, x)[0]

        # cross-attention
        x = self.norm2(x)
        x = x + self.cross_attn(x, encoder_output, encoder_output)[0]

        # feed-forward network
        x = x + self.ffm(self.norm3(x))
        return x


class VisionTransformerDecoder(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embed_dim,
        n_heads,
        hidden_dim,
        n_layers,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        # positional embedding of shape (1, n_patches, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))

        # how many layers of transformer encoder to stack
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(embed_dim, n_heads, hidden_dim, dropout)
                for _ in range(n_layers)
            ]
        )

        # Final layer to map back to original patch dimension
        self.output_projection = nn.Linear(
            embed_dim, self.img_size * self.img_size * self.in_channels
        )

    def forward(self, encoding):
        B, n_patches, embed_dim = encoding.shape
        assert (
            embed_dim == self.embed_dim
        ), f"Embedding dimension should match the encoding: {embed_dim} != {self.embed_dim}"

        # (B, n_patches, embed_dim)
        # If positional embeddings are used, add them (optional)
        x = encoding + self.pos_embed[:, :n_patches, :]  # Positional embedding addition

        for layer in self.decoder_layers:
            x = x + layer(x, encoding)

        # Map back to original patch dimension (unflatten the patches)
        # (B, n_patches, embed_dim) -> (B, n_patches, img_size * img_size * in_channels)
        patches = self.output_projection(x)
        patches = patches.view(B, n_patches, self.img_size, self.img_size, self.in_channels)
        patches = patches.permute(0, 4, 2, 3, 1)  # [B, 3, img_size, img_size, num_patches]

        # Reconstruct the original image by reshaping patches
        img = patches.reshape(B, n_patches, self.in_channels, self.img_size, self.img_size)
        return img  # Reconstructed image

    # def forward(self, encoding):
    #     B, n_patches, embed_dim = encoding.shape
    #     assert (
    #         embed_dim == self.embed_dim
    #     ), f"Expected embed_dim={self.embed_dim}, but got {embed_dim}"

    #     # If positional embeddings are used, add them (optional)
    #     x = encoding + self.pos_embed[:, :n_patches, :]  # Positional embedding addition

    #     # Pass through Transformer decoder layers
    #     for layer in self.decoder_layers:
    #         x = layer(
    #             x, encoding
    #         )  # Optionally use the encoding as context for the transformer decoder

    #     # Flatten batches and patches into a single dimension (Batches * Patches, Embed_dim)
    #     x = x.view(B * n_patches, embed_dim)

    #     # Project from `embed_dim` back to original patch size (Batches * Patches, Patch Size)
    #     patches = self.output_projection(x)

    #     # Reconstruct the original image from patches
    #     img = patches.reshape(
    #         B * n_patches, self.in_channels, self.img_size, self.img_size
    #     )

    #     return img  # Reconstructed image
