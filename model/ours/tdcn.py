# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self,
                 in_features,
                 filter_widths,
                 out_features):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.in_features = in_features
        self.filter_widths = filter_widths
        self.out_features = out_features
        self.pad = [filter_widths[0] // 2]

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 3  # batch, sequence, feature
        assert x.shape[-1] == self.in_features

        x = self._forward_blocks(x.permute(0, 2, 1))

        return x.permute(0, 2, 1)


class TDCN(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self,
                 in_features,
                 filter_widths,
                 causal,
                 dropout,
                 channels,
                 out_features,
                 dense):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(in_features,
                         filter_widths,
                         out_features)

        self.repeat_len = (filter_widths[0] ** len(filter_widths)) // 2

        self.expand_conv = nn.Sequential(
            nn.Conv1d(in_features, channels[0], filter_widths[0], bias=False),
            nn.BatchNorm1d(channels[0], momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )

        layers_conv = []
        res_conv = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            layers_conv.append(
                nn.Sequential(
                    nn.Conv1d(channels[i-1], channels[i],
                              filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                              dilation=next_dilation if not dense else 1, bias=False),
                    nn.BatchNorm1d(channels[i], momentum=0.1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    nn.Conv1d(channels[i], channels[i], 1, dilation=1, bias=False),
                    nn.BatchNorm1d(channels[i], momentum=0.1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ))

            res_conv.append(
                nn.Conv1d(channels[i-1], channels[i], 1)
            )

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.res_conv = nn.ModuleList(res_conv)
        self.shrink = nn.Conv1d(channels[-1], self.out_features, 1)

    def _forward_blocks(self, x):
        x = self.expand_conv(x)

        for pad, shift, layer, rlayer in zip(self.pad[1:], self.causal_shift[1:], self.layers_conv, self.res_conv):
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]
            x = rlayer(res) + layer(x)

        return self.shrink(x)
