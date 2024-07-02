import torch
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
from torch import nn
from torch.nn import functional as F


class VAE_Encoder(nn.Module):
    def __init__(self):
        super(VAE_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.residual1 = VAE_ResidualBlock(in_channels=128, out_channels=128)
        self.residual2 = VAE_ResidualBlock(in_channels=128, out_channels=128)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0
        )
        self.residual3 = VAE_ResidualBlock(in_channels=128, out_channels=256)
        self.residual4 = VAE_ResidualBlock(in_channels=256, out_channels=256)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0
        )
        self.residual5 = VAE_ResidualBlock(in_channels=256, out_channels=512)
        self.residual6 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.conv4 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0
        )
        self.residual7 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.residual8 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.residual9 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.attention1 = VAE_AttentionBlock(in_channels=512)
        self.residual10 = VAE_ResidualBlock(in_channels=512, out_channels=512)
        self.grpnorm1 = nn.GroupNorm(num_groups=32, num_channels=512)
        self.silu1 = nn.SiLU()
        self.conv5 = nn.Conv2d(
            in_channels=512, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, noise):
        # input (x): (Batch_Size, 3, Height, Width)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)

        # (Batch_Size, 3, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.conv1(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.residual1(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.residual2(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.conv2(x)
        # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        x = self.residual3(x)
        # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        x = self.residual4(x)
        # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.conv3(x)
        # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.residual5(x)
        # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.residual6(x)
        # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.conv4(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.residual7(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.residual8(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.residual9(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.attention1(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.residual10(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.grpnorm1(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.silu1(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
        x = self.conv5(x)
        # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
        x = self.conv6(x)

        # Get mean and log_variance of the distribution learned by the VAE
        # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8) X 2
        mean, log_variance = torch.chunk(x, chunks=2, dim=1)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, min=-30.0, max=20.0)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        std = variance.sqrt()

        # Sample from the distribution (= Map noise, a stardard normal distribution, to the mean and variance learned by VAE)
        # Transform noise from N(0, 1) -> N(mean, stdev)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + std * noise

        # Scale by a constant
        x *= 0.18215

        return x


print(torch.__version__)
print(torch.cuda.is_available())
