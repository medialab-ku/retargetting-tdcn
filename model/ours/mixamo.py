import pickle
import random

import numpy as np
from torch.utils.data import Dataset
from core.quaternions import Quaternions
import core.anim.bvh as BVH
import core.anim.animation as Animation

joints_list = ["Spine", "Spine1", "Spine2", "Neck", "Head", "LeftUpLeg",
               "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg",
               "RightLeg", "RightFoot", "RightToeBase", "LeftShoulder",
               "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder",
               "RightArm", "RightForeArm", "RightHand"]


class MIXAMOdataset(Dataset):

    def __init__(self,
                 binary_path,
                 out_frame,
                 scale_augment=True,
                 rot_augment=False,
                 scale_std=0.1,
                 num_critics=5,
                 dtype=np.float32):
        with open(binary_path, 'rb') as binary:
            train_dataset = pickle.load(binary)

        self.train = train_dataset['dataset']
        self.out_frame = out_frame
        self.scale_augment = scale_augment
        self.rot_augment = rot_augment
        self.scale_std = scale_std
        self.dtype = dtype
        self.num_critics = num_critics

        self.f = train_dataset['f']
        self.bvh_scale = train_dataset['scale']

        self.anim = list()
        for c, a, path in self.train:
            anim, _, _ = BVH.load(path)

            if anim.positions.shape[0] <= 1:
                raise ValueError("anim.position has not enough frame! : {} frame".format(anim.positions.shape[0]))

            with open(path) as fp:
                bvh_file = fp.read().split("JOINT")
                bvh_joints = [line.split("\n")[0] for line in bvh_file[1:]]
                to_keep = [0]
                for jname in joints_list:
                    for l in range(len(bvh_joints)):
                        if jname == bvh_joints[l][-len(jname):]:
                            to_keep.append(l + 1)
                            break

            anim.parents = anim.parents[to_keep]
            for i in range(1, len(anim.parents)):
                """ If joint not needed, connect to the previous joint """
                if anim.parents[i] not in to_keep:
                    anim.parents[i] = anim.parents[i] - 1
                anim.parents[i] = to_keep.index(anim.parents[i])

            anim.positions = anim.positions[:, to_keep, :]
            anim.rotations.qs = anim.rotations.qs[:, to_keep, :]
            anim.orients.qs = anim.orients.qs[to_keep, :]
            anim.eulers = anim.eulers[:, to_keep, :]

            self.anim.append(anim)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idxA):
        (localA, offsetA, skelA, quatA, maskA,
         localB, offsetB, skelB, quatB, maskB) = self.get_pair(idxA)

        critic_out = list()
        for i in range(self.num_critics):
            idxR = np.random.randint(self.__len__())
            critic_out.append(self.get_pair(idxR))

        return (localA, offsetA, skelA, quatA, maskA,
                localB, offsetB, skelB, quatB, maskB,
                critic_out)

    def get_pair(self, idxA):
        maskA = np.zeros(shape=self.out_frame, dtype=self.dtype)
        maskB = np.zeros(shape=self.out_frame, dtype=self.dtype)

        scale = self.bvh_scale
        aug_q = Quaternions.id(1)

        if self.scale_augment:
            t = random.gauss(1, self.scale_std)
            t = max(t, 0.1)
            scale *= t

        if self.rot_augment:
            rot = random.random()
            rot = (rot - 0.5) * 2 * np.pi

            aug_q = Quaternions.from_angle_axis(rot, np.array([0, 1, 0])) * aug_q

        quatA, localA, rtjA, skelA = self.get_normalized_data(idxA, scale, aug_q)
        random_slice = get_random_slice(localA, self.out_frame)
        maskA[:np.min([self.out_frame, localA.shape[0]])] = 1.0
        localA = repeat_last_frame(localA[random_slice], self.out_frame)
        rtjA = repeat_last_frame(rtjA[random_slice], self.out_frame)
        skelA = repeat_last_frame(skelA[random_slice], self.out_frame)
        quatA = repeat_last_frame(quatA[random_slice], self.out_frame)

        # get real skeleton and motion
        skelA_name, _, _ = self.train[idxA]
        idxB = np.random.randint(self.__len__())
        while skelA_name == self.train[idxB][0]:
            idxB = np.random.randint(self.__len__())

        quatB, localB, rtjB, skelB = self.get_normalized_data(idxB, scale, aug_q)
        random_slice = get_random_slice(localB, self.out_frame)
        maskB[:np.min([self.out_frame, localB.shape[0]])] = 1.0
        localB = repeat_last_frame(localB[random_slice], self.out_frame)
        rtjB = repeat_last_frame(rtjB[random_slice], self.out_frame)
        skelB = repeat_last_frame(skelB[random_slice], self.out_frame)
        quatB = repeat_last_frame(quatB[random_slice], self.out_frame)

        localA = localA.astype(self.dtype)
        rtjA = rtjA.astype(self.dtype)
        skelA = skelA.astype(self.dtype)
        quatA = quatA.astype(self.dtype)

        localB = localB.astype(self.dtype)
        rtjB = rtjB.astype(self.dtype)
        skelB = skelB.astype(self.dtype)
        quatB = quatB.astype(self.dtype)

        return (localA, rtjA, skelA, quatA, maskA,
                localB, rtjB, skelB, quatB, maskB)

    def get_normalized_data(self, index, scale, rot):
        anim = self.anim[index].copy()
        anim.positions = anim.positions * scale

        anim.rotations[:, 0] = rot * anim.rotations[:, 0]
        quat = anim.rotations.qs.copy()
        local = Animation.positions_local(anim.copy())

        anim.positions[:, 0] = rot * anim.positions[:, 0]
        rtj = anim.positions[:, 0].copy()

        anim.rotations.qs[...] = anim.orients.qs[None]  # get t-pose
        tjoints = Animation.positions_local(anim)
        anim.positions[...] = get_skel(tjoints[0], anim.parents)[None]
        skel = anim.positions.copy()

        # resize domain
        local = (local - self.f['local']['mean']) / self.f['local']['std']
        rtj = (rtj - self.f['rtj']['mean']) / self.f['rtj']['std']
        skel = (skel - self.f['local']['mean']) / self.f['local']['std']

        return quat, local, rtj, skel


def get_foot_contacts(anim):
    positions = Animation.positions_global(anim.copy())

    """ Put on Floor """
    fid_l, fid_r = np.array([8, 9]), np.array([12, 13])
    foot_heights = np.minimum(positions[:, fid_l, 1],
                              positions[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)

    positions[:, :, 1] -= floor_height

    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.15, 0.15]), np.array([9.0, 6.0])

    feet_l_len = np.sum((positions[1:, fid_l] - positions[:-1, fid_l]) ** 2, axis=-1)
    feet_l_h = positions[:-1, fid_l, 1]
    feet_l = ((feet_l_len < velfactor) & (feet_l_h < heightfactor))

    feet_r_len = np.sum((positions[1:, fid_r] - positions[:-1, fid_r]) ** 2, axis=-1)
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = ((feet_r_len < velfactor) & (feet_r_h < heightfactor))

    foot_contacts = np.concatenate((feet_l, feet_r), axis=-1)
    foot_contacts = np.concatenate((foot_contacts, foot_contacts[-1:]), axis=0)

    return foot_contacts, floor_height


def repeat_last_frame(array, max_step):
    length = array.shape[0]
    if length < max_step:
        last_array = array[length - 1: length]
        repeat = np.ones(len(array.shape), dtype=int)
        repeat[0] = max_step - length
        last_array = np.tile(last_array, repeat)
        return np.concatenate((array, last_array))
    else:
        return array


def get_random_slice(array, max_step):
    lower_bound = 0
    upper_bound = array.shape[0] - max_step
    if lower_bound >= upper_bound:
        st_idx = 0
    else:
        st_idx = np.random.randint(low=lower_bound, high=upper_bound)
    ed_idx = st_idx + max_step
    rnd_slice = slice(st_idx, ed_idx)

    return rnd_slice


def get_skel(joints, parents):
    c_offsets = []
    for j in range(parents.shape[0]):
        if parents[j] != -1:
            c_offsets.append(joints[j, :] - joints[parents[j], :])
        else:
            c_offsets.append(joints[j, :])
    return np.stack(c_offsets, axis=0)


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)
