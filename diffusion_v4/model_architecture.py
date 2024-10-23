import torch
import torch.nn as nn
import torch.nn.functional as F

## Model Architecture
class SelfAttention(nn.Module):
    """
    Capture long range dependecies -> Research more
    """
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    Down sampling to extract high-level features
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.conv_down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.double_conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.conv_down(x)  # Downsample from 5x5 to 3x3 using convolution
        x = self.double_conv(x)  # Apply the double convolution layers
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Expand and repeat timestep embedding
        return x + emb  # Combine feature map with embedding

class Up(nn.Module):
    """
    Upsampling
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        # up needs to be adjusted to go from 3x3 to 5x5
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up = nn.Upsample(size=(5, 5), mode="bilinear", align_corners=True)

        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def __init__(self, c_in=1, c_out=1, time_dim=256, device=device):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16) # Check out
        self.down1 = Down(16, 16)
        self.sa1 = SelfAttention(16, 3)

        self.bot1 = DoubleConv(16, 32)  # Bottleneck layers
        self.bot2 = DoubleConv(32, 32)
        self.bot3 = DoubleConv(32, 16)

        self.up1 = Up(32, 16)  # Single upsampling layer
        self.sa2 = SelfAttention(16, 5)  # Self-Attention after upsampling, with size back to 5x5
        self.outc = nn.Conv2d(16, c_out, kernel_size=1)  # Final convolution layer


    def pos_encoding(self, t, channels):
            inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
            pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
            pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
            pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
            return pos_enc




    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)

        x2 = self.bot1(x2)
        x2 = self.bot2(x2)
        x2 = self.bot3(x2)

        x = self.up1(x2, x1, t)
        x = self.sa2(x)
        output = self.outc(x)
        return output

# diff_model = UNet()
#
# print("Num params: ", sum(p.numel() for p in diff_model.parameters()))