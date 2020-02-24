"""Based on Daniel Holden code from
http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing"""

import torch
import torch.nn as nn


class ForwardKinematics(nn.Module):
    def __init__(self):
        super(ForwardKinematics, self).__init__()

    def transforms_blank(self, rotations):
        diagonal = torch.eye(4, dtype=rotations.dtype, device=rotations.device)
        diagonal = diagonal[None, None, :, :]
        ts = diagonal.repeat(rotations.shape[0], rotations.shape[1], 1, 1)

        return ts

    def transforms_rotations(self, rotations):
        q_length = rotations.norm(dim=-1)
        q_unit = rotations / q_length[..., None]
        qw, qx, qy, qz = q_unit[..., 0], q_unit[..., 1], q_unit[..., 2], q_unit[..., 3]
        """Unit quaternion based rotation matrix computation"""
        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1)
        dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1)
        dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1)
        m = torch.stack([dim0, dim1, dim2], dim=-2)

        return m

    def transforms_local(self, positions, rotations):
        transforms = self.transforms_rotations(rotations)  # (B, J, 3, 3)
        transforms = torch.cat([transforms, positions.unsqueeze(-1)], dim=-1)  # (B, J, 3, 4]

        zeros = torch.zeros([transforms.shape[0], transforms.shape[1], 1, 3], dtype=rotations.dtype,
                            device=rotations.device)
        ones = torch.ones([transforms.shape[0], transforms.shape[1], 1, 1], dtype=rotations.dtype,
                          device=rotations.device)
        zerosones = torch.cat([zeros, ones], dim=-1)  # (B, J, 1, 4]
        transforms = torch.cat([transforms, zerosones], dim=-2)  # (B J 4 4)
        return transforms

    def transforms_global(self, parents, positions, rotations):
        locals_ = self.transforms_local(positions, rotations)  # (B J 4 4)
        globals_ = self.transforms_blank(rotations)  # (B J 4 4)

        globals_ = torch.cat([locals_[:, 0:1], globals_[:, 1:]], dim=1)  # (B J 4 4)
        for jt in range(1, len(parents)):
            globals_[:, jt] = torch.matmul(globals_[:, parents[jt]], locals_[:, jt])
        return globals_

    def forward(self, parents, positions, rotations):
        positions = self.transforms_global(parents, positions, rotations)[:, :, :, 3]
        return positions[:, :, :3] / positions[:, :, 3, None]
