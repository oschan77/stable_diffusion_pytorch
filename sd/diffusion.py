import torch
from attention import CrossAttention, SelfAttention
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super(TimeEmbedding, self).__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear1(x)
        # (1, 1280) -> (1, 1280)
        x = F.silu(x)
        # (1, 1280) -> (1, 1280)
        x = self.linear2(x)

        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)

        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super(UNET_ResidualBlock, self).__init__()
        self.grpnorm_feature = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.linear_time = nn.Linear(n_time, out_channels)
        self.grpnorm_merged = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_merged = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Current Height, Current Width)
        # time: (1, 1280)
        # out: (Batch_Size, Out_Channels, Current Height, Current Width)

        # residue: (Batch_Size, Out_Channels, Current Height, Current Width)
        residue = feature

        # (Batch_Size, In_Channels, Current Height, Current Width) -> (Batch_Size, In_Channels, Current Height, Current Width)
        feature = self.grpnorm_feature(feature)
        # (Batch_Size, In_Channels, Current Height, Current Width) -> (Batch_Size, In_Channels, Current Height, Current Width)
        feature = F.silu(feature)
        # (Batch_Size, In_Channels, Current Height, Current Width) -> (Batch_Size, Out_Channels, Current Height, Current Width)
        feature = self.conv_feature(feature)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        # (1, Out_Channels) -> (1, Out_Channels, 1, 1)
        time = time.unsqueeze(-1).unsqueeze(-1)

        # (Batch_Size, Out_Channels, Current Height, Current Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Current Height, Current Width)
        merged = feature + time
        # (Batch_Size, Out_Channels, Current Height, Current Width) -> (Batch_Size, Out_Channels, Current Height, Current Width)
        merged = self.grpnorm_merged(merged)
        # (Batch_Size, Out_Channels, Current Height, Current Width) -> (Batch_Size, Out_Channels, Current Height, Current Width)
        merged = F.silu(merged)
        # (Batch_Size, Out_Channels, Current Height, Current Width) -> (Batch_Size, Out_Channels, Current Height, Current Width)
        merged = self.conv_merged(merged)

        # (Batch_Size, In_Channels, Current Height, Current Width) -> (Batch_Size, Out_Channels, Current Height, Current Width)
        residue = self.residual_layer(residue)

        # (Batch_Size, Out_Channels, Current Height, Current Width) + (Batch_Size, Out_Channels, Current Height, Current Width) -> (Batch_Size, Out_Channels, Current Height, Current Width)
        out = merged + residue

        return out


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embd, d_context=1280):
        super(UNET_AttentionBlock, self).__init__()
        channels = n_embd * n_head
        self.grpnorm1 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.layernorm1 = nn.LayerNorm(channels)
        self.self_attention1 = SelfAttention(
            n_head=n_head,
            n_embd=channels,
            in_proj_bias=False,
            out_proj_bias=True,
        )
        self.layernorm2 = nn.LayerNorm(channels)
        self.cross_attention = CrossAttention(
            n_head=n_head,
            n_embd=channels,
            d_context=d_context,
            in_proj_bias=False,
            out_proj_bias=True,
        )
        self.layernorm3 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, 4 * channels * 2)
        self.linear2 = nn.Linear(4 * channels, channels)
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, context):
        # x: (Batch_Size, Channels, Current Height, Current Width)
        # context: (Batch_Size, Seq_Len, Dim)

        # Preprocess image features into sequence for attention mechanism
        # residue_long: (Batch_Size, Channels, Current Height, Current Width)
        residue_long = x
        # (Batch_Size, Channels, Current Height, Current Width) -> (Batch_Size, Channels, Current Height, Current Width)
        x = self.grpnorm1(x)
        # (Batch_Size, Channels, Current Height, Current Width) -> (Batch_Size, Channels, Current Height, Current Width)
        x = self.conv1(x)
        b, c, h, w = x.shape
        # (Batch_Size, Channels, Current Height, Current Width) -> (Batch_Size, Channels, Current Height * Current Width)
        x = x.view(b, c, h * w)
        # (Batch_Size, Channels, Current Height * Current Width) -> (Batch_Size, Current Height * Current Width, Channels)
        x = x.transpose(-1, -2)

        # Normalization + Self-Attention + Skip Connection
        # residue_short: (Batch_Size, Current Height * Current Width, Channels)
        residue_short = x
        # (Batch_Size, Current Height * Current Width, Channels) -> (Batch_Size, Current Height * Current Width, Channels)
        x = self.layernorm1(x)
        # (Batch_Size, Current Height * Current Width, Channels) -> (Batch_Size, Current Height * Current Width, Channels)
        x = self.self_attention1(x)
        # (Batch_Size, Current Height * Current Width, Channels) -> (Batch_Size, Current Height * Current Width, Channels)
        x = x + residue_short

        # Normalization + Cross-Attention + Skip Connection
        # residue_short: (Batch_Size, Current Height * Current Width, Channels)
        residue_short = x
        # (Batch_Size, Current Height * Current Width, Channels) -> (Batch_Size, Current Height * Current Width, Channels)
        x = self.layernorm2(x)
        # (Batch_Size, Current Height * Current Width, Channels) -> (Batch_Size, Current Height * Current Width, Channels)
        x = self.cross_attention(x, context)
        # (Batch_Size, Current Height * Current Width, Channels) -> (Batch_Size, Current Height * Current Width, Channels)
        x = x + residue_short

        # Normalization + MLP with GeGLU + Skip Connection
        # residue_short: (Batch_Size, Current Height * Current Width, Channels)
        residue_short = x
        # (Batch_Size, Current Height * Current Width, Channels) -> (Batch_Size, Current Height * Current Width, Channels)
        x = self.layernorm3(x)
        # (Batch_Size, Current Height * Current Width, Channels) -> (Batch_Size, Current Height * Current Width, 4 * Channels * 2)
        x = self.linear1(x)
        # (Batch_Size, Current Height * Current Width, 4 * Channels * 2) -> (Batch_Size, Current Height * Current Width, 4 * Channels) X 2
        x, gate = x.chunk(2, dim=-1)
        # (Batch_Size, Current Height * Current Width, 4 * Channels) -> (Batch_Size, Current Height * Current Width, 4 * Channels)
        x = x * F.gelu(gate)
        # (Batch_Size, Current Height * Current Width, 4 * Channels) -> (Batch_Size, Current Height * Current Width, Channels)
        x = self.linear2(x)
        # (Batch_Size, Current Height * Current Width, Channels) -> (Batch_Size, Current Height * Current Width, Channels)
        x = x + residue_short

        # Postprocess sequence back into image features
        # (Batch_Size, Current Height * Current Width, Channels) -> (Batch_Size, Channels, Current Height * Current Width)
        x = x.transpose(-1, -2)
        # (Batch_Size, Channels, Current Height * Current Width) -> (Batch_Size, Channels, Current Height, Current Width)
        x = x.view(b, c, h, w)

        # Conv Layer + (Initial) Skip Connection
        # (Batch_Size, Channels, Current Height, Current Width) -> (Batch_Size, Channels, Current Height, Current Width)
        x = self.conv2(x)
        # (Batch_Size, Channels, Current Height, Current Width) -> (Batch_Size, Channels, Current Height, Current Width)
        x = x + residue_long

        return x


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.encoders = nn.ModuleList(
            [
                # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    nn.Conv2d(
                        in_channels=4,
                        out_channels=320,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ),
                # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=320, out_channels=320),
                    UNET_AttentionBlock(n_head=8, n_embd=40),
                ),
                # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=320, out_channels=320),
                    UNET_AttentionBlock(n_head=8, n_embd=40),
                ),
                # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
                SwitchSequential(
                    nn.Conv2d(
                        in_channels=320,
                        out_channels=320,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ),
                # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=320, out_channels=640),
                    UNET_AttentionBlock(n_head=8, n_embd=80),
                ),
                # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=640, out_channels=640),
                    UNET_AttentionBlock(n_head=8, n_embd=80),
                ),
                # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
                SwitchSequential(
                    nn.Conv2d(
                        in_channels=640,
                        out_channels=640,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ),
                # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=640, out_channels=1280),
                    UNET_AttentionBlock(n_head=8, n_embd=160),
                ),
                # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1280, out_channels=1280),
                    UNET_AttentionBlock(n_head=8, n_embd=160),
                ),
                # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(
                    nn.Conv2d(
                        in_channels=1280,
                        out_channels=1280,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ),
                # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1280, out_channels=1280),
                ),
                # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1280, out_channels=1280),
                ),
            ]
        )

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(in_channels=1280, out_channels=1280),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(n_head=8, n_embd=160),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(in_channels=1280, out_channels=1280),
        )

        # channels in decoder = channels of embeddings + skip connections saved in encoder
        self.decoders = nn.ModuleList(
            [
                # (Batch_Size, 2560 (1280 + 1280), Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=2560, out_channels=1280),
                ),
                # (Batch_Size, 2560 (1280 + 1280), Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=2560, out_channels=1280),
                ),
                # (Batch_Size, 2560 (1280 + 1280), Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=2560, out_channels=1280),
                    Upsample(channels=1280),
                ),
                # (Batch_Size, 2560 (1280 + 1280), Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=2560, out_channels=1280),
                    UNET_AttentionBlock(n_head=8, n_embd=160),
                ),
                # (Batch_Size, 2560 (1280 + 1280), Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=2560, out_channels=1280),
                    UNET_AttentionBlock(n_head=8, n_embd=160),
                ),
                # (Batch_Size, 1920 (1280 + 640), Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1920, out_channels=1280),
                    UNET_AttentionBlock(n_head=8, n_embd=160),
                    Upsample(channels=1280),
                ),
                # (Batch_Size, 1920 (1280 + 640), Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1920, out_channels=640),
                    UNET_AttentionBlock(n_head=8, n_embd=80),
                ),
                # (Batch_Size, 1280 (640 + 640), Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1280, out_channels=640),
                    UNET_AttentionBlock(n_head=8, n_embd=80),
                ),
                # (Batch_Size, 1280 (640 + 320), Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=960, out_channels=640),
                    UNET_AttentionBlock(n_head=8, n_embd=80),
                    Upsample(channels=640),
                ),
                # (Batch_Size, 960 (640 + 320), Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=960, out_channels=320),
                    UNET_AttentionBlock(n_head=8, n_embd=40),
                ),
                # (Batch_Size, 640 (320 + 320), Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=640, out_channels=320),
                    UNET_AttentionBlock(n_head=8, n_embd=40),
                ),
                # (Batch_Size, 640 (320 + 320), Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=640, out_channels=320),
                    UNET_AttentionBlock(n_head=8, n_embd=40),
                ),
            ]
        )

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 1280)
        # out: (Batch_Size, 320, Height / 8, Width / 8)

        skip_connections = []
        for layer in self.encoders:
            x = layer(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layer in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x, context, time)

        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNET_OutputLayer, self).__init__()
        self.grpnorm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.grpnorm1(x)
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv1(x)

        return x


class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()
        self.time_embedding = TimeEmbedding(n_embd=320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(in_channels=320, out_channels=4)

    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)
        # out (Required by Decoder) : (Batch_Size, 4, Height / 8, Width / 8)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8), (Batch, Seq_Len, Dim), (1, 1280) -> (Batch, 320, Height / 8, Width / 8)
        out = self.unet(latent, context, time)
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        out = self.final(out)

        return out
