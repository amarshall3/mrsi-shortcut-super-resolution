# Standard UNet Model, forward pass updated to embed "dt" term
# Add "TimeConditioner" module for non-linear combination of time and delta time embeddings

# Code modified from Huggingface unet_2d:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d.py



from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import (
    get_down_block,
    get_mid_block,
    get_up_block,
    UNetMidBlock2D,
)

@dataclass
class UNet2DOutput(BaseOutput):
    sample: torch.Tensor


class TimeConditioner(nn.Module):
    #  combines t and dt embeddings with mlp for shortcut step size conditioning
    def __init__(self, time_embed_dim):
        super().__init__()
        self.combine = nn.Sequential(
            nn.Linear(time_embed_dim * 2, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
    
    def forward(self, t_emb, dt_emb):
        # t_emb, dt_emb: [B, time_embed_dim]
        concatenated = torch.cat([t_emb, dt_emb], dim=1)
        return self.combine(concatenated)


class UNet2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        mid_block_type: Optional[str] = "UNetMidBlock2D",
        up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        handle_delta_time: bool = True,
        num_train_timesteps: Optional[int] = 1000,
        add_attention: bool = True,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # time embedding
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        elif time_embedding_type == "learned":
            self.time_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # delta time embedding (added for shortuct model)
        if handle_delta_time:
            if time_embedding_type == "fourier":
                self.dt_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
                dt_input_dim = 2 * block_out_channels[0]
            elif time_embedding_type == "positional":
                self.dt_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
                dt_input_dim = block_out_channels[0]
            elif time_embedding_type == "learned":
                self.dt_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
                dt_input_dim = block_out_channels[0]

            self.dt_embedding = TimestepEmbedding(dt_input_dim, time_embed_dim)
            
            # time conditioner for non-linear t + dt interaction (added for shortcut model)
            self.time_conditioner = TimeConditioner(time_embed_dim)
        else:
            self.dt_proj = None
            self.dt_embedding = None
            self.time_conditioner = None

        # down blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid block
        if mid_block_type is not None:
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
                resnet_groups=norm_num_groups,
                attn_groups=attn_norm_num_groups,
                add_attention=add_attention,
            )
        else:
            self.mid_block = None

        # up blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # output
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        delta_time: Optional[Union[torch.Tensor, float, int]] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        Forward pass for the UNet2DModel.

        Args:
            sample (`torch.Tensor`): The input tensor with shape `(batch_size, in_channels, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The timestep for conditioning.
            delta_time (`torch.Tensor` or `float` or `int`, optional): The delta time for conditioning.
            return_dict (`bool`, optional): Whether to return a `UNet2DOutput` or a tuple.

        Returns:
            `UNet2DOutput` or `tuple`: The output tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time embeddings
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # handle delta time embedding (added for shortcut model)
        if self.dt_proj is not None and delta_time is not None:
            if not torch.is_tensor(delta_time):
                delta_time = torch.tensor([delta_time], dtype=torch.long, device=sample.device)
            elif torch.is_tensor(delta_time) and len(delta_time.shape) == 0:
                delta_time = delta_time[None].to(sample.device)

            # broadcast to batch dimension
            delta_time = delta_time * torch.ones(sample.shape[0], dtype=delta_time.dtype, device=delta_time.device)

            dt_emb = self.dt_proj(delta_time)
            dt_emb = dt_emb.to(dtype=self.dtype)
            dt_emb = self.dt_embedding(dt_emb)

            #  use time_conditioner for non-linear combination of t and dt
            emb = self.time_conditioner(emb, dt_emb)

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down blocks
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid block
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        # 5. up blocks
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)