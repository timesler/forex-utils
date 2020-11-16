import torch
from torch import nn


class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=self.padding, 
            dilation=dilation, 
            **kwargs
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.padding]
        return x


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, dilation, norm=False):
        super().__init__()
        self.conv_in = nn.Sequential(
            CausalConv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        )
        self.conv_res = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.conv_skip = nn.Conv1d(out_channels, skip_channels, 1, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_in = x
        x = self.conv_in(x)
        x = self.tanh(x) * self.sigmoid(x)
        skip = self.conv_skip(x)
        x = self.conv_res(x)
        x = x + x_in

        return x, skip


class TimNet(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        in_length,
        skip_length,
        res_channels=16,
        dilation_factor=2,
        norm=False,
        pool=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_length = in_length
        self.skip_length = skip_length
        self.norm = norm
        self.pool = pool

        self.conv_in = nn.Conv1d(in_channels, res_channels, 1, bias=False)

        receptive_field = 1
        dilation = 1
        self.res_blocks = nn.ModuleList()
        while receptive_field < in_length:
            self.res_blocks.append(
                GatedResidualBlock(
                    in_channels=res_channels,
                    out_channels=res_channels,
                    skip_channels=res_channels,
                    kernel_size=2,
                    dilation=dilation,
                    norm=norm
                )
            )
            receptive_field += dilation
            dilation *= dilation_factor
        
        self.receptive_field = receptive_field
        self.dilation = dilation

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) if pool else nn.Identity(),
            nn.Flatten(),
            nn.Linear(
                res_channels * 1 if pool else res_channels * skip_length,
                out_channels * 1 if pool else out_channels * skip_length,
                bias=False
            )
        )
    
    def forward(self, x):
        x = self.conv_in(x)
        skips = 0
        for mod in self.res_blocks:
            x, skip = mod(x)
            skips = skips + skip
        x = skips[:, :, -self.skip_length:]
        x = self.out(x)
        if not self.pool:
            x = x.view(-1, self.out_channels, self.skip_length)
        return x
