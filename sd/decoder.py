from attention import SelfAttention
from torch import nn


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VAE_ResidualBlock, self).__init__()
        self.grpnorm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.silu1 = nn.SiLU()
        self.grpnorm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.silu2 = nn.SiLU()

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

    def forward(self, x):
        # input (x): (Batch_Size, In_Channels, Height, Width)
        # residue: (Batch_Size, In_Channels, Height, Width)
        residue = x

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = self.grpnorm1(x)
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv1(x)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.silu1(x)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.grpnorm2(x)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv2(x)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.silu2(x)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = x + self.residual_layer(residue)

        return x


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(VAE_AttentionBlock, self).__init__()
        self.grpnorm1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(n_heads=1, d_embed=channels)

    def forward(self, x):
        # input (x): (Batch_Size, Channels, Height, Width)
        # residue: (Batch_Size, In_Channels, Height, Width)
        residue = x

        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height, Width)
        x = self.grpnorm1(x)

        b, c, h, w = x.shape
        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height * Width)
        x = x.view(b, c, h * w)
        # (Batch_Size, Channels, Height * Width) -> (Batch_Size, Height * Width, Channels)
        x = x.transpose(-1, -2)
        # (Batch_Size, Height * Width, Channels) -> (Batch_Size, Height * Width, Channels)
        x = self.attention(x)
        # (Batch_Size, Height * Width, Channels) -> (Batch_Size, Channels, Height * Width)
        x = x.transpose(-1, -2)
        # (Batch_Size, Channels, Height * Width) -> (Batch_Size, Channels, Height, Width)
        x = x.view(b, c, h, w)
        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height, Width)
        x = x + residue

        return x


class VAE_Decoder(nn.Module):
    def __init__(self):
        super(VAE_Decoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=4, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.residual1 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.attention1 = VAE_AttentionBlock(in_channels=512)
        self.residual2 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.residual3 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.residual4 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.residual5 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.residual6 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.residual7 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.residual8 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.residual9 = VAE_ResidualBlock(in_channels=512, out_channels=256)
        self.residual10 = VAE_ResidualBlock(in_channels=256, out_channels=256)
        self.residual11 = VAE_ResidualBlock(in_channels=256, out_channels=256)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.residual12 = VAE_ResidualBlock(in_channels=256, out_channels=128)
        self.residual13 = VAE_ResidualBlock(in_channels=128, out_channels=128)
        self.residual14 = VAE_ResidualBlock(in_channels=128, out_channels=128)
        self.grpnorm1 = nn.GroupNorm(num_groups=32, num_channels=128)
        self.silu1 = nn.SiLU()
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        x = x / 0.18215

        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv1(x)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.conv2(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.residual1(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.attention1(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.residual2(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.residual3(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.residual4(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.residual5(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.upsample1(x)
        # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.conv3(x)
        # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.residual6(x)
        # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.residual7(x)
        # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.residual8(x)
        # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
        x = self.upsample2(x)
        # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 2, Width / 2)
        x = self.conv4(x)
        # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        x = self.residual9(x)
        # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        x = self.residual10(x)
        # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        x = self.residual11(x)
        # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
        x = self.upsample3(x)
        # (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width)
        x = self.conv5(x)
        # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.residual12(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.residual13(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.residual14(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.grpnorm1(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.silu1(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
        x = self.conv6(x)

        return x
