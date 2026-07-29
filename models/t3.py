"""
A wrapper for the T3-tiny DIGIT tactile encoder.

Adapted from the official T3 implementation and pretrained weights:
https://github.com/alanzjl/t3
https://huggingface.co/datasets/alanz-mit/FoundationTactile

Reference:
Zhao, J., Ma, Y., Wang, L., & Adelson, E. (2025). Transferable Tactile
Transformers for Representation Learning Across Diverse Sensors and Tasks.
Proceedings of The 8th Conference on Robot Learning, PMLR 270, 3766-3779.

Author: Gemma McLean
Date: July 2026
"""

from pathlib import Path
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from timm.models.vision_transformer import Block, PatchEmbed


# T3-tiny pretrained model repository and revision on Hugging Face
T3_REPOSITORY = "alanz-mit/FoundationTactile"
T3_REVISION = "e1a16123575eb26e789cf0129ced6f3ba081f2ed"
T3_MODEL_NAME = "t3_tiny"


class T3TinyBackbone(nn.Module):
    """T3-tiny DIGIT encoder and shared transformer trunk."""

    # The feature dimension of the T3-tiny model is 192
    feature_dim = 192

    def __init__(self, checkpoint_dir=None):
        """
        Initialise the T3-tiny backbone and load the pretrained DIGIT encoder weights.

        Args:
            checkpoint_dir (str or Path, optional): Folder containing the downloaded T3
                checkpoint files. Defaults to None.
        """

        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "digit": T3Encoder(),
                "trunk": T3Trunk(),
            }
        )
        self._load_pretrained_weights(checkpoint_dir)

    def _load_pretrained_weights(self, checkpoint_dir):
        """
        Load the DIGIT encoder and shared trunk weights from Hugging Face.

        Args:
            checkpoint_dir (str or Path, optional): Folder containing the downloaded T3
                checkpoint files. Defaults to None.
        """

        # Load pretrained weights from Hugging Face if no checkpoint directory is provided
        if checkpoint_dir is None:
            encoder_path = hf_hub_download(
                repo_id=T3_REPOSITORY,
                repo_type="dataset",
                filename=f"models/{T3_MODEL_NAME}/encoders/digit.pth",
                revision=T3_REVISION,
            )
            trunk_path = hf_hub_download(
                repo_id=T3_REPOSITORY,
                repo_type="dataset",
                filename=f"models/{T3_MODEL_NAME}/trunk.pth",
                revision=T3_REVISION,
            )
        # Otherwise, load pretrained weights from the provided checkpoint directory
        else:
            checkpoint_dir = Path(checkpoint_dir)
            encoder_path = checkpoint_dir / "encoders" / "digit.pth"
            trunk_path = checkpoint_dir / "trunk.pth"

        # Load the pretrained weights into the encoder and trunk
        self.encoder["digit"].load_state_dict(
            torch.load(encoder_path, map_location="cpu", weights_only=True)
        )
        self.encoder["trunk"].load_state_dict(
            torch.load(trunk_path, map_location="cpu", weights_only=True)
        )

    def forward(self, x, return_features=False):
        """
        Return T3 features for a batch of DIGIT images.

        Args:
            x (torch.Tensor): A batch of DIGIT images with shape (batch_size, 3, 224, 224).
            return_features (bool): If True, return the features before the classifier
                instead of the final output.
        """

        # Pass the input through the DIGIT encoder and shared trunk to get features
        features = self.encoder["trunk"](self.encoder["digit"](x))[:, 0]
        return features


class T3Encoder(nn.Module):
    """The 3-block ViT encoder used by T3-tiny for DIGIT images."""

    def __init__(self):
        """Initialise the T3-tiny DIGIT encoder architecture."""

        super().__init__()

        # Define the patch embedding layer
        self.patch_embed = PatchEmbed(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=T3TinyBackbone.feature_dim,
        )
        # Define the class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, T3TinyBackbone.feature_dim))

        # Define the positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 197, T3TinyBackbone.feature_dim), requires_grad=False
        )

        # Define the 3 transformer blocks for the encoder
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=T3TinyBackbone.feature_dim,
                    num_heads=3,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(3)
            ]
        )

    def forward(self, x):
        """
        Convert a batch of images into T3 token embeddings.

        Args:
            x (torch.Tensor): A batch of images with shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: A batch of token embeddings with shape (batch_size, 197,
                feature_dim).
        """

        # Apply the patch embedding, add the class token and positional embedding
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        # Pass through the 3 transformer blocks
        for block in self.blocks:
            x = block(x)
        return x


class T3Trunk(nn.Module):
    """The 9-block shared transformer trunk used by T3-tiny."""

    def __init__(self):
        """Initialise the T3-tiny shared transformer trunk architecture."""

        super().__init__()

        # Define the 9 transformer blocks for the trunk
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=T3TinyBackbone.feature_dim,
                    num_heads=3,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(9)
            ]
        )

        # Define the final layer normalisation for the trunk
        self.norm = nn.LayerNorm(T3TinyBackbone.feature_dim)

    def forward(self, x):
        """
        Apply the shared T3 transformer blocks to token embeddings.

        Args:
            x (torch.Tensor): A batch of token embeddings with shape (batch_size, 197,
                feature_dim).

        Returns:
            torch.Tensor: A batch of token embeddings with shape (batch_size, 197,
                feature_dim) after passing through the shared transformer blocks
                and final layer normalisation.
        """

        # Pass through the 9 transformer blocks and apply final layer normalisation
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
