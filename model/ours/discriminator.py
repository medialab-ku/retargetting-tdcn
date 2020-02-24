import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,
                 in_features,
                 filter_widths,
                 channels,
                 leak,
                 causal=False,
                 dense=False):
        super(Discriminator, self).__init__()

        self.repeat_len = (filter_widths[0] ** len(filter_widths)) // 2

        self.in_features = in_features
        self.filter_widths = filter_widths
        self.channels = channels
        self.leak = leak
        self.causal = causal
        self.dense = dense

        self.pad = [filter_widths[0] // 2]

        self.init_conv = nn.Sequential(
            nn.Conv1d(in_features, channels, filter_widths[0], bias=False),
            nn.InstanceNorm1d(channels),
            nn.LeakyReLU(leak),
        )

        layers_conv = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            layers_conv.append(nn.Sequential(
                nn.Conv1d(channels, channels,
                          filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                          padding=0,
                          dilation=next_dilation if not dense else 1,
                          bias=False),
                nn.InstanceNorm1d(channels),
                nn.LeakyReLU(leak),
                nn.Conv1d(channels, channels, 1, dilation=1, padding=0, bias=False),
                nn.InstanceNorm1d(channels),
                nn.LeakyReLU(leak),
            ))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)

        self.shrink = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        assert len(x.shape) == 3
        assert x.shape[-1] == self.in_features

        x = x.permute(0, 2, 1)
        x = self.init_conv(x)

        for pad, shift, layer in zip(self.pad[1:], self.causal_shift[1:], self.layers_conv):
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]
            x = res + layer(x)

        x = self.shrink(x)

        return x.squeeze()
