from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from core.abstract.trainer import Trainer
from utils.utils import qmul


class OursTrainer(Trainer):
    def __init__(self, config, model, optim, writer):
        super(OursTrainer, self).__init__(config, model, optim, writer)

        f = self.data_loader.dataset.f
        self.local_mean = torch.from_numpy(f['local']['mean']).to(dtype=self.config.dtype, device=self.config.device)
        self.local_std = torch.from_numpy(f['local']['std']).to(dtype=self.config.dtype, device=self.config.device)
        self.rtj_mean = torch.from_numpy(f['rtj']['mean']).to(dtype=self.config.dtype, device=self.config.device)
        self.rtj_std = torch.from_numpy(f['rtj']['std']).to(dtype=self.config.dtype, device=self.config.device)

        rads = np.radians(self.config.angle)
        self.rads = torch.tensor(rads, dtype=self.config.dtype, device=self.config.device, requires_grad=False)
        self.unit_q = torch.tensor([1, 0, 0, 0],
                                   dtype=self.config.dtype, device=self.config.device, requires_grad=False)
        self.unit_q = self.unit_q.reshape(1, 1, 4)

        self.mse = nn.MSELoss(reduction='none')
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')
        self.post_score = 1

    def step(self, epoch, global_step):
        postfix = OrderedDict({
            'G': 0,
            'D': 0,
            'curr_score': 0,
        })

        self.model.G.train()
        self.model.D.train()

        epoch = epoch + 1
        avg_G_loss = 0
        avg_D_loss = 0

        with tqdm(total=len(self.data_loader), desc='{} epoch'.format(epoch)) as progress:
            for (localA, rtjA, skelA, quatA, maskA,
                 localB, rtjB, skelB, quatB, maskB,
                 critic_data) in self.data_loader:
                gen_data = {
                    'localXA': localA.to(self.config.device),
                    'rtjXA': rtjA.to(self.config.device),
                    'skelA': skelA[:, 0:1, ...].to(self.config.device),
                    'quatXA': quatA.to(self.config.device),
                    'maskX': maskA.to(dtype=self.config.dtype, device=self.config.device),

                    'localYB': localB.to(self.config.device),
                    'rtjYB': rtjB.to(self.config.device),
                    'skelB': skelB[:, 0:1, ...].to(self.config.device),
                    'quatYB': quatB.to(self.config.device),
                    'maskY': maskB.to(dtype=self.config.dtype, device=self.config.device),

                    'rads': self.rads,
                    'step': global_step,
                    'epoch': epoch,
                }

                if self.post_score > self.config.advScore:
                    D_loss = self.train_discriminator(critic_data, global_step)
                    avg_D_loss += D_loss
                    postfix['D'] = D_loss

                total_loss, current_score = self.train_generator(**gen_data)
                avg_G_loss += total_loss.item()

                global_step = global_step + 1
                postfix['total'] = total_loss.item()
                postfix['G'] = total_loss.item()
                postfix['curr_score'] = current_score.item()
                if self.config.advScore is not None:
                    self.post_score = 0.99 * self.post_score + 0.01 * current_score.item()
                progress.set_postfix(ordered_dict=postfix)

                progress.update(1)

        avg_G_loss /= len(self.data_loader)

        return epoch, global_step, avg_G_loss

    def step_no_grad(self):
        pass

    def train_generator(self,
                        localXA, rtjXA, skelA, quatXA, maskX,
                        localYB, rtjYB, skelB, quatYB, maskY,
                        rads, step, epoch,
                        **kwargs):
        summarys = dict()
        EIDX = self.config.endeffectors
        self.optim.G.zero_grad()
        divisorX = torch.sum(maskX)
        total_loss = 0

        # RETARGETTING  X: AA, AAA, ABA
        encXA, outputs = self.model.G(quatXA, rtjXA, skelA, [skelA, skelB])
        localXAA, dlocalXAA, rtjXAA, quatXAA, raquatXAA = outputs[0]
        localXAB, dlocalXAB, rtjXAB, quatXAB, raquatXAB = outputs[1]

        _, (localXAAA, dlocalXAAA, rtjXAAA, quatXAAA, raquatXAAA) = self.model.G(quatXAA, rtjXAA, skelA, skelA)
        encXAB, (localXABA, dlocalXABA, rtjXABA, quatXABA, raquatXABA) = self.model.G(quatXAB, rtjXAB, skelB, skelA)

        # autoencoder loss
        autoX_local = torch.sum(maskX[..., None, None] * self.mse(localXAA[:, :, EIDX], localXA[:, :, EIDX])) / divisorX
        autoX_rtj = torch.sum(maskX[..., None] * self.mse(rtjXAA, rtjXA)) / divisorX
        autoX_quat = torch.sum(maskX[..., None, None] * self.mse(quatXAA, quatXA)) / divisorX

        autoX_loss = self.config.localW * autoX_local + autoX_rtj + self.config.quatW * autoX_quat
        auto_loss = autoX_loss  # + autoY_loss
        total_loss = total_loss + self.config.autoW * auto_loss

        summarys['loss/autoX_loss'] = autoX_loss
        summarys['loss/autoX_local'] = autoX_local
        summarys['loss/autoX_rtj'] = autoX_rtj
        summarys['loss/autoX_quat'] = autoX_quat

        # cycle loss
        cycleXABA_local = torch.sum(
            maskX[..., None, None] * self.mse(localXABA[:, :, EIDX], localXA[:, :, EIDX])) / divisorX
        cycleXABA_rtj = torch.sum(maskX[..., None] * self.mse(rtjXABA, rtjXA)) / divisorX
        cycleXABA_quat = torch.sum(maskX[..., None, None] * self.mse(quatXABA, quatXA)) / divisorX

        cycleXAAA_local = torch.sum(
            maskX[..., None, None] * self.mse(localXAAA[:, :, EIDX], localXA[:, :, EIDX])) / divisorX
        cycleXAAA_rtj = torch.sum(maskX[..., None] * self.mse(rtjXAAA, rtjXA)) / divisorX
        cycleXAAA_quat = torch.sum(maskX[..., None, None] * self.mse(quatXAAA, quatXA)) / divisorX

        cycleXABA_loss = self.config.localW * cycleXABA_local + cycleXABA_rtj + self.config.quatW * cycleXABA_quat
        cycleXAAA_loss = self.config.localW * cycleXAAA_local + cycleXAAA_rtj + self.config.quatW * cycleXAAA_quat
        cycleX_loss = cycleXABA_loss + cycleXAAA_loss

        cycle_loss = cycleX_loss  # + cycleY_loss
        total_loss = total_loss + self.config.cycleW * cycle_loss

        summarys['loss/cycleXABA_loss'] = cycleXABA_loss
        summarys['loss/cycleXABA_local'] = cycleXABA_local
        summarys['loss/cycleXABA_rtj'] = cycleXABA_rtj
        summarys['loss/cycleXABA_quat'] = cycleXABA_quat

        summarys['loss/cycleXAAA_loss'] = cycleXAAA_loss
        summarys['loss/cycleXAAA_local'] = cycleXAAA_local
        summarys['loss/cycleXAAA_rtj'] = cycleXAAA_rtj
        summarys['loss/cycleXAAA_quat'] = cycleXAAA_quat

        summarys['loss/cycleX_loss'] = cycleX_loss

        # height loss
        dskelA = skelA * self.local_std + self.local_mean
        dskelB = skelB * self.local_std + self.local_mean

        dlocalXA = localXA * self.local_std + self.local_mean
        drtjXA = rtjXA * self.rtj_std + self.rtj_mean
        drtjXAB = rtjXAB * self.rtj_std + self.rtj_mean
        diffX_local, diffX_rtj = self.get_scaled_loss(dlocalXA, drtjXA, dskelA, dlocalXAB, drtjXAB, dskelB)

        heightX_local = torch.sum(maskX[:, 1:, None] * diffX_local[:, :, EIDX]) / divisorX
        heightX_rtj = torch.sum(maskX[:, 1:] * diffX_rtj) / divisorX
        heightX_loss = self.config.localW * heightX_local + self.config.rtjW * heightX_rtj
        height_loss = heightX_loss
        total_loss = total_loss + self.config.heightW * height_loss

        summarys['loss/heightX_loss'] = heightX_loss
        summarys['loss/heightX_local'] = heightX_local
        summarys['loss/heightX_rtj'] = heightX_rtj

        # oriantation loss
        euler_y = maskX * get_euler_y(qmul(quatXAB[:, :, 0], -quatXA[:, :, 0]))
        orientation_loss = torch.sum(
            self.smoothl1(euler_y, torch.zeros_like(euler_y))) / divisorX
        total_loss = total_loss + self.config.orientationW * orientation_loss
        summarys['loss/orientation_loss'] = orientation_loss

        # twist loss
        quatsX = torch.cat([quatXAA, quatXAB, quatXAAA, quatXABA], dim=0)[:, :, 1:]
        angleX = maskX.repeat(4, 1)[..., None] * torch.abs(get_euler_angle(quatsX))
        twistX_loss = torch.sum(torch.max(torch.zeros_like(angleX), angleX - rads) ** 2) / divisorX
        twist_loss = twistX_loss  # + twistY_loss
        total_loss = total_loss + self.config.twistW * twist_loss

        summarys['loss/twistX_loss'] = twistX_loss

        # quat normlization
        raquatsX = torch.cat([raquatXAA, raquatXAB, raquatXAAA, raquatXABA], dim=0)
        raquatX_len = raquatsX.norm(dim=-1)
        raquatX_loss = torch.sum(
            maskX.repeat(4, 1)[..., None] * self.mse(raquatX_len, torch.ones_like(raquatX_len))) / divisorX
        raquat_loss = raquatX_loss
        total_loss = total_loss + self.config.raquatW * raquat_loss

        summarys['loss/raquatX_loss'] = raquatX_loss

        # adversarial loss
        # make 'FAKE' data
        fake_data = self.get_disc_input(dlocalXAB[:, :, EIDX], drtjXAB, dskelB)
        fake_score = self.model.D(fake_data)

        fake_loss = self.mse(fake_score, torch.ones_like(fake_score))
        adv_loss = torch.sum(maskX[:, :-1] * fake_loss) / divisorX
        total_loss = total_loss + self.config.advW * adv_loss

        summarys['loss/adv_loss'] = adv_loss

        # total loss
        summarys['loss/total_loss'] = total_loss

        total_loss.backward()
        self.optim.G.step()

        for k, v in summarys.items():
            self.writer.add_scalar(k, v, step)

        # debug
        if step % 100 == 0:
            for name, param in self.model.G.named_parameters():
                target = ['weight', 'bias']
                if any(x in name.split('.')[-1] for x in target):
                    self.writer.add_histogram(name, param, step)
                    self.writer.add_histogram('grad.' + name, param.grad, step)

        return total_loss, fake_score.mean()

    def train_discriminator(self, critic_data, step):

        real_scores = list()
        fake_scores = list()

        D_ = list()

        for data in critic_data:
            (localXA, rtjXA, skelA, quatXA, maskX,
             localZC, rtjZC, skelC, quatZC, maskZ) = data

            rtjXA = rtjXA.to(self.config.device)
            skelA = skelA[:, 0:1, ...].to(self.config.device)
            quatXA = quatXA.to(self.config.device)
            maskX = maskX.to(dtype=self.config.dtype, device=self.config.device)
            divisorX = torch.sum(maskX)

            localZC = localZC.to(self.config.device)
            rtjZC = rtjZC.to(self.config.device)
            skelC = skelC[:, 0:1, ...].to(self.config.device)  # (batch, 1, joint, 3)
            maskZ = maskZ.to(dtype=self.config.dtype, device=self.config.device)
            divisorZ = torch.sum(maskZ)

            self.optim.D.zero_grad()
            EIDX = self.config.endeffectors

            _, (localXAC, dlocalXAC, rtjXAC, _, _) = self.model.G(quatXA, rtjXA, skelA, skelC)

            # make 'REAL' data
            dlocalZC = localZC * self.local_std + self.local_mean
            drtjZC = rtjZC * self.rtj_std + self.rtj_mean
            dskelC = skelC * self.local_std + self.local_mean
            real_data = self.get_disc_input(dlocalZC[:, :, EIDX], drtjZC, dskelC)

            # make 'FAKE' data
            drtjXAC = rtjXAC * self.rtj_std + self.rtj_mean
            fake_data = self.get_disc_input(dlocalXAC[:, :, EIDX], drtjXAC, dskelC)

            real_score = self.model.D(real_data)  # B, F-1
            fake_score = self.model.D(fake_data)  # B, F-1

            real_loss = self.mse(real_score, torch.ones_like(real_score))
            fake_loss = self.mse(fake_score, torch.zeros_like(fake_score))

            real_loss = torch.sum(maskZ[:, :-1] * real_loss) / divisorZ
            fake_loss = torch.sum(maskX[:, :-1] * fake_loss) / divisorX

            D_loss = real_loss + fake_loss
            D_loss.backward()

            self.optim.D.step()

            real_scores.append(real_loss.item())
            fake_scores.append(fake_loss.item())
            D_.append(D_loss.item())

        real_score = sum(real_scores) / len(real_scores)
        fake_score = sum(fake_scores) / len(fake_scores)
        D_loss = sum(D_) / len(D_)

        self.writer.add_scalar('train/D/real_score', real_score, step)
        self.writer.add_scalar('train/D/fake_score', fake_score, step)
        self.writer.add_scalar('train/D', D_loss, step)

        return D_loss

    def get_disc_input(self, dlocal, drtj, dskel):
        batch, frame = dlocal.shape[0:2]
        height_jts = [5, 4, 3, 2, 1, 7, 8, 9]
        bone_len = dskel[:, 0:1].norm(dim=-1)
        height = bone_len[:, 0, height_jts].sum(dim=-1)
        bone_len = bone_len.repeat(1, frame - 1, 1)

        diff_local = dlocal[:, 1:] - dlocal[:, :-1]
        diff_len_local = diff_local.norm(dim=-1) / height[:, None, None]

        diff_len_rtj = (drtj[:, 1:] - drtj[:, :-1]).norm(dim=-1) / height[:, None]
        disc_input = torch.cat([diff_len_local, diff_len_rtj[..., None], bone_len], dim=-1)
        disc_input = self.repeat_sequence(disc_input, self.model.D.repeat_len, self.model.D.causal)

        return disc_input

    def repeat_sequence(self, seq, repeat_len, causal=False):
        if causal:
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

    def get_scaled_loss(self, dlocalA, drtjA, dskelA, dlocalB, drtjB, dskelB):
        heightA = get_skel_height(dskelA)
        heightB = get_skel_height(dskelB)

        diff_localA = dlocalA[:, 1:] - dlocalA[:, :-1]
        diff_len_localA = diff_localA.norm(dim=-1) / heightA[:, None, None]

        diff_localB = dlocalB[:, 1:] - dlocalB[:, :-1]
        diff_len_localB = diff_localB.norm(dim=-1) / heightB[:, None, None]

        diff_len_rtjA = drtjA[:, :-1].norm(dim=-1) / heightA[:, None]
        diff_len_rtjB = drtjB[:, :-1].norm(dim=-1) / heightB[:, None]

        diff_local = self.smoothl1(diff_len_localA, diff_len_localB)
        diff_rtj = self.smoothl1(diff_len_rtjA, diff_len_rtjB)

        return diff_local, diff_rtj


def get_euler_angle(quats, order="yzx"):
    q0 = quats[..., 0]
    q1 = quats[..., 1]
    q2 = quats[..., 2]
    q3 = quats[..., 3]

    if order == "xyz":
        ex = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        ey = torch.asin(torch.clamp(2 * (q0 * q2 - q3 * q1), -1, 1))
        ez = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        return torch.stack([ex, ez], dim=-1)  # [:, :, 1:]
    elif order == "yzx":
        ex = torch.atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
        ey = torch.atan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
        ez = torch.asin(torch.clamp(2 * (q1 * q2 + q3 * q0), -1, 1))
        return ey  # [:, :, 1:]
    else:
        raise Exception("Unknown Euler order!")


def get_euler_y(quats):
    q0 = quats[..., 0]
    q1 = quats[..., 1]
    q2 = quats[..., 2]
    q3 = quats[..., 3]

    ey = torch.atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
    return ey  # [:, :, 1:]


def get_height(joints):
    jts1 = [5, 4, 3, 2, 1, 7, 8, 9]
    jts2 = [4, 3, 2, 1, 0, 6, 7, 8]
    return (joints[:, 0, jts1] - joints[:, 0, jts2]).norm(dim=-1).sum(dim=-1)


def get_skel_height(skel):
    jts = [5, 4, 3, 2, 1, 7, 8, 9]
    return skel[:, 0, jts].norm(dim=-1).sum(dim=-1)
