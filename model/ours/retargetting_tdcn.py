import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

from ops.forward_kinematics import ForwardKinematics
from .tdcn import TDCN


class RetargettingTDCN(nn.Module):

    def __init__(self,
                 n_joints,
                 filter_widths,
                 dropout,
                 channels,
                 out_features,
                 causal,
                 dense,
                 parents,
                 binary_path,
                 ):
        super(RetargettingTDCN, self).__init__()

        dtype = torch.get_default_dtype()

        self.n_joints = n_joints

        self.filter_widths = filter_widths
        self.dropout = dropout
        self.channels = channels
        self.out_features = out_features

        self.causal = causal
        self.dense = dense
        self.parents = parents

        self.angle_tm = TDCN(in_features=(n_joints * 4) + 3 + 2 * (n_joints * 3),  # 3d coordinate and offset
                             filter_widths=filter_widths,
                             causal=causal,
                             dropout=dropout,
                             channels=channels,
                             out_features=out_features,
                             dense=dense)

        self.qlinear = RootLinear(out_features, self.n_joints)  # no infer root rotation just use it.
        self.forward_kinematics = ForwardKinematics()

        with open(binary_path, 'rb') as binary:
            f = pickle.load(binary)['f']

        self.quat_id = nn.Parameter(torch.tensor([1, 0, 0, 0], dtype=dtype).view(1, 1, 4), requires_grad=False)
        self.local_mean = nn.Parameter(torch.from_numpy(f['local']['mean']).to(dtype), requires_grad=False)
        self.local_std = nn.Parameter(torch.from_numpy(f['local']['std']).to(dtype), requires_grad=False)
        self.rtj_mean = nn.Parameter(torch.from_numpy(f['rtj']['mean']).to(dtype), requires_grad=False)
        self.rtj_std = nn.Parameter(torch.from_numpy(f['rtj']['std']).to(dtype), requires_grad=False)

    def repeat_sequence(self, seq, repeat_len):
        if self.causal:
            first_seq = seq[:, 0, ...].unsqueeze(1)
            first_seq = first_seq.repeat(1, 2 * repeat_len, 1)
            seq = torch.cat([first_seq, seq], dim=1)
        else:
            first_seq = seq[:, 0, ...].unsqueeze(1)
            last_seq = seq[:, -1, ...].unsqueeze(1)
            first_seq = first_seq.repeat(1, repeat_len, 1)
            last_seq = last_seq.repeat(1, repeat_len, 1)
            seq = torch.cat([first_seq, seq, last_seq], dim=1)

        return seq

    def forward(self, quatA, rtjA, skelA, skels):
        batch, frame = quatA.shape[0:2]

        quat_id = self.quat_id.repeat(batch * frame, 1, 1)
        fk_quatA = torch.cat([quat_id, quatA[:, :, 1:].view(batch * frame, self.n_joints - 1, 4)], dim=1)
        denorm_skelA = skelA * self.local_std + self.local_mean
        denorm_skelA = denorm_skelA.repeat(1, frame, 1, 1).view(-1, self.n_joints, 3)
        denorm_localA = self.forward_kinematics(self.parents, denorm_skelA, fk_quatA)
        localA = (denorm_localA - self.local_mean) / self.local_std
        seqA = torch.cat([rtjA, localA.view(batch, frame, -1)], dim=-1)

        if not isinstance(skels, list):
            skels = [skels]

        outputs = list()
        for skel in skels:
            localB, dlocalB, rtjB, quatB, raw_quatB = self.retargeting(quatA, seqA, skel, batch, frame)
            outputs.append((localB, dlocalB, rtjB, quatB, raw_quatB))

        if len(outputs) == 1:
            outputs = outputs[0]

        return None, outputs

    def retargeting(self, quatA, seqA, skelB, batch, frame):  # sourceA, targetB
        denorm_skelB = skelB * self.local_std + self.local_mean
        denorm_skelB = denorm_skelB.repeat(1, frame, 1, 1).view(-1, self.n_joints, 3)

        skelB = skelB.view(batch, 1, -1).repeat(1, frame, 1)
        quatAwithSkelB = torch.cat([quatA.view(batch, frame, -1), skelB, seqA], dim=-1)
        repeated_quatAwithSkelB = self.repeat_sequence(quatAwithSkelB, self.angle_tm.repeat_len)
        angleB = self.angle_tm(repeated_quatAwithSkelB)
        angleB = angleB.contiguous().view(-1, self.out_features)
        quatBwithRootB = self.qlinear(angleB)
        raw_quatB = quatBwithRootB[:, :-3]
        rtjB = quatBwithRootB[:, -3:]

        raw_quatB = raw_quatB.view(batch * frame, self.n_joints, 4)

        quat_id = self.quat_id.repeat(batch * frame, 1, 1)
        fk_quatB = torch.cat([quat_id, raw_quatB[:, 1:]], dim=1)

        denorm_localB = self.forward_kinematics(self.parents, denorm_skelB, fk_quatB)
        localB = (denorm_localB - self.local_mean) / self.local_std

        localB = localB.view(batch, frame, self.n_joints, 3)
        denorm_localB = denorm_localB.view(batch, frame, self.n_joints, 3)

        rtjB = rtjB.view(batch, frame, 3)
        raw_quatB = raw_quatB.view(batch, frame, self.n_joints, 4)
        norm_quatB = raw_quatB / raw_quatB.norm(dim=-1)[..., None]

        return localB, denorm_localB, rtjB, norm_quatB, raw_quatB


class RootLinear(nn.Module):
    def __init__(self,
                 in_features,
                 n_joints,
                 bias=True,
                 init_std=0.002):
        super(RootLinear, self).__init__()
        self.in_features = in_features
        self.n_joints = n_joints
        self.init_std = init_std
        # do not infer quat. of the root joint
        self.weight = nn.Parameter(torch.Tensor(n_joints * 4 + 3, in_features))
        if bias:
            bias_init = np.tile(np.array([1, 0, 0, 0]), n_joints)
            bias_init = np.concatenate([bias_init, np.array([0, 0, 0])])
            self.bias = nn.Parameter(torch.Tensor(bias_init))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight, std=self.init_std)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, n_joints={}, bias={}'.format(
            self.in_features, self.n_joints, self.bias is not None)
