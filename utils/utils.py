import os
from operator import itemgetter

import numpy as np
import torch
from parse import parse

import core.anim.animation as Animation
import core.anim.bvh as BVH
from model.ours.mixamo import get_skel


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def put_in_world(local, rtj, rootrot, diff_y=None):
    if diff_y is not None:
        rtj[:, 1] += diff_y
    local[:, :, 1] -= local[:, 0:1, 1]  # remove y-offset

    rtj = rtj[:, 0:3].copy()
    world = rootrot[:, None] * local + rtj[:, None]

    return world


def put_in_world_batch(dlocal, drtj, rootrots, diff_y=None):
    batch, frames, jt = dlocal.shape[0:3]
    drtj = drtj[:, :, 0:3]

    if diff_y is not None:
        drtj[:, :, 1] += diff_y
    dlocal[:, :, :, 1] -= dlocal[:, :, 0:1, 1]

    world = qrot(rootrots[:, :, None].repeat(1, 1, jt, 1), dlocal) + drtj[:, :, None]

    return world


def load_testdata_root(min_steps, max_steps, scale, testset='nkn'):
    if testset is 'nkn':
        data_path = "./datasets/test/"
        km_kc = [
            ("known_motion/Kaya/", "known_character/Warrok_W_Kurniawan1/"),
            ("known_motion/Big_Vegas/", "known_character/Malcolm1/"),
        ]
        km_nc = {
            ("known_motion/AJ/", "new_character/Mutant1/"),
            ("known_motion/Peasant_Man/", "new_character/Liam1/")
        }
        nm_kc = {
            ("new_motion/Granny/", "known_character/Malcolm2/"),
            ("new_motion/Claire/", "known_character/Warrok_W_Kurniawan2/")
        }
        nm_nc = {
            ("new_motion/Mutant/", "new_character/Liam2/"),
            ("new_motion/Claire/", "new_character/Mutant2/")
        }
    else:
        raise ValueError("target ({}) is not valid!".format(testset))

    test_list = [km_kc, km_nc, nm_kc, nm_nc]

    inpjoints = []
    tgtjoints = []

    inpanims = []
    tgtanims = []
    gtanims = []

    inpquats = []
    tgtquats = []

    inplocals = []
    tgtlocals = []

    inprtjs = []
    tgtrtjs = []

    inpskels = []
    tgtskels = []

    from_names = []
    to_names = []

    for test_item in test_list:
        for inp, tgt in test_item:
            files = sorted([
                f for f in os.listdir(data_path + inp)
                if not f.startswith(".") and f.endswith(".bvh")
            ])
            for cfile in files:
                inpanim, inpnames, inpftime = BVH.load(data_path + inp + "/" + cfile)
                tgtanim, tgtnames, tgtftime = BVH.load(data_path + tgt + "/" + cfile)

                if inpanim.shape[0] <= min_steps:
                    # print("Validation: skip {}. ({} frame)".format(cfile, inpanim.shape[0]))
                    continue

                inpanim.positions = inpanim.positions * scale
                inpanim.offsets = inpanim.offsets * scale

                tgtanim.positions = tgtanim.positions * scale
                tgtanim.offsets = tgtanim.offsets * scale

                gtanim = tgtanim.copy()

                inppath = data_path + inp + "/" + cfile
                tgtpath = data_path + tgt + "/" + cfile

                inpquat, inplocal, inprtj, inpskel, ito_keep = get_test_data(inpanim, inppath)
                tgtquat, tgtlocal, tgtrtj, tgtskel, tto_keep = get_test_data(tgtanim, tgtpath)

                num_samples = 1
                idx_fn = lambda idx: slice(None)
                if max_steps is not None:
                    num_samples = inplocal.shape[0] // max_steps
                    idx_fn = lambda idx: slice(idx * max_steps, (idx + 1) * max_steps)

                for s in range(num_samples):
                    inpjoints.append(ito_keep)
                    tgtjoints.append(tto_keep)
                    inpanims.append([
                        inpanim.copy()[idx_fn(s)],
                        inpnames,
                        inpftime
                    ])
                    tgtanims.append([
                        tgtanim.copy()[idx_fn(s)],
                        tgtnames,
                        tgtftime
                    ])
                    gtanims.append([
                        gtanim.copy()[idx_fn(s)],
                        tgtnames,
                        tgtftime
                    ])

                    inpquats.append(inpquat[idx_fn(s)])
                    tgtquats.append(tgtquat[idx_fn(s)])

                    inplocals.append(inplocal[idx_fn(s)])
                    tgtlocals.append(tgtlocal[idx_fn(s)])

                    inprtjs.append(inprtj[idx_fn(s)])
                    tgtrtjs.append(tgtrtj[idx_fn(s)])

                    inpskels.append(inpskel)
                    tgtskels.append(tgtskel)

                    from_names.append(inp.split("/")[0] + "_" + inp.split("/")[1])
                    to_names.append(tgt.split("/")[0] + "_" + tgt.split("/")[1])

                if max_steps is not None and not inplocal.shape[0] % max_steps == 0:
                    inpjoints.append(ito_keep)
                    tgtjoints.append(tto_keep)
                    inpanims.append([
                        inpanim.copy()[-max_steps:],
                        inpnames,
                        inpftime
                    ])
                    tgtanims.append([
                        tgtanim.copy()[-max_steps:],
                        tgtnames,
                        tgtftime
                    ])
                    gtanims.append([
                        gtanim.copy()[-max_steps:],
                        tgtnames,
                        tgtftime
                    ])

                    inpquats.append(inpquat[-max_steps:])
                    tgtquats.append(tgtquat[-max_steps:])

                    inplocals.append(inplocal[-max_steps:])
                    tgtlocals.append(tgtlocal[-max_steps:])

                    inprtjs.append(inprtj[-max_steps:])
                    tgtrtjs.append(tgtrtj[-max_steps:])

                    inpskels.append(inpskel)
                    tgtskels.append(tgtskel)

                    from_names.append(inp.split("/")[0] + "_" + inp.split("/")[1])
                    to_names.append(tgt.split("/")[0] + "_" + tgt.split("/")[1])

    return (inpjoints, inpanims, inpquats, inplocals, inprtjs, inpskels, from_names,
            tgtjoints, tgtanims, tgtquats, tgtlocals, tgtrtjs, tgtskels, to_names,
            gtanims)


def get_test_data(animation, file_path):
    anim = animation.copy()

    joints_list = [
        "Spine", "Spine1", "Spine2", "Neck", "Head", "LeftUpLeg", "LeftLeg",
        "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot",
        "RightToeBase", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand"
    ]

    with open(file_path) as fp:
        bvh_file = fp.read().split("JOINT")
    bvh_joints = [f.split("\n")[0].split(":")[-1].split(" ")[-1] for f in bvh_file[1:]]
    to_keep = [0]
    for jname in joints_list:
        for k in range(len(bvh_joints)):
            if jname == bvh_joints[k][-len(jname):]:
                to_keep.append(k + 1)
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

    quat = anim.rotations.qs.copy()
    local = Animation.positions_local(anim.copy())

    # foot_contacts, feet_height = get_foot_contacts(anim)
    rtj = anim.positions[:, 0].copy()

    anim.rotations.qs[...] = anim.orients.qs[None]  # get t-pose
    tjoints = Animation.positions_local(anim)
    anim.positions[...] = get_skel(tjoints[0], anim.parents)[None]
    skel = anim.positions.copy()[0:1]

    return quat, local, rtj, skel, to_keep


def copy_model(cfg, epoch):
    path = os.path.join('.', 'ckpt', *cfg.split('.'))
    gen_param_path = os.path.join(path, 'gen', 'param')
    eval_param_path = os.path.join(path, 'eval', 'param')

    if not os.path.exists(gen_param_path):
        raise ValueError("path does not exist!:" + gen_param_path)

    from core.saver import CKPT_FMT
    target = 'best.ckpt'
    if epoch is None:
        if not os.path.isfile(os.path.join(gen_param_path, 'best.ckpt')):
            raise ValueError("The best ckeckpoint doen not exist. Please check the config and the model.")
    else:
        if not os.path.isfile(os.path.join(gen_param_path, CKPT_FMT.format(epoch))):
            raise ValueError("The target epoch doen not exist. Please check the config and the target epoch.")
        target = CKPT_FMT.format(epoch)

    if not os.path.exists(eval_param_path):
        os.makedirs(eval_param_path)

    import shutil
    shutil.copy(os.path.join(gen_param_path, target),
                os.path.join(eval_param_path, '0001.ckpt'))


def get_skel_height_np(skel):
    return (np.sqrt(((skel[:, 0, 5, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((skel[:, 0, 4, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((skel[:, 0, 3, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((skel[:, 0, 2, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((skel[:, 0, 1, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((skel[:, 0, 7, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((skel[:, 0, 8, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((skel[:, 0, 9, :]) ** 2).sum(axis=-1)))


def get_bonelengths(joints):
    parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15,
                        16, 3, 18, 19, 20])
    c_offsets = []
    for j in range(parents.shape[0]):
        if parents[j] != -1:
            c_offsets.append(joints[0, j, :] - joints[0, parents[j], :])
        else:
            c_offsets.append(joints[0, j, :])
    offsets = np.stack(c_offsets, axis=0)
    return np.sqrt(((offsets) ** 2).sum(axis=-1))[..., 1:]


def compare_bls(bl1, bl2):
    relbones = np.array([-1, 0, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, -1, 13, 14,
                         15, -1, 17, 18, 19])

    bl_diff = np.abs(bl1 - bl2).mean()

    bl1ratios = []
    bl2ratios = []
    for j in range(len(relbones)):
        if relbones[j] != -1:
            bl1ratios.append(bl1[j] / bl1[relbones[j]])
            bl2ratios.append(bl2[j] / bl2[relbones[j]])

    blratios_diff = np.abs(np.stack(bl1ratios) - np.stack(bl2ratios)).mean()

    return bl_diff, blratios_diff


def create_report(diff_worlds, diff_locals, diff_rtjs, vels,
                  labels, bl_diffs, blratio_diffs, inpheights, tgtheights,
                  suffix, target_path):
    report_file = os.path.join(target_path, "..", "report_{}".format(suffix))
    avgs = list()
    scale = 5.625
    with open(report_file, "w") as fp:
        for label in ["known_motion/known_character",
                      "known_motion/new_character",
                      "new_motion/known_character",
                      "new_motion/new_character", ]:
            fp.write("-" * 55 + "\n")
            fp.write("| {:38}{:^10}\n".format(label.upper() + ".", " MEAN"))
            fp.write("-" * 55 + "\n")
            idxs = [i for i, j in enumerate(labels)
                    if label.split("/")[0] in j.split("/")[0] and
                    label.split("/")[1] in j.split("/")[1]]
            fp.write("| Number of Examples: {}.\n".format(len(idxs)))
            if idxs:
                _mean = np.concatenate(itemgetter(*idxs)(diff_worlds), axis=0).mean()
                fp.write("| AVG WORLD                          {:>10.4f}\n".format(_mean / scale))
                avgs.append(_mean / scale)

                fp.write("-" * 55 + "\n")

                _mean = np.array(inpheights)[idxs].mean()
                fp.write("| INPUT  CHARACTER HEIGHT            {:>10.4f}\n".format(_mean / scale))

                _mean = np.array(tgtheights)[idxs].mean()
                fp.write("| TARGET CHARACTER HEIGHT            {:>10.4f}\n".format(_mean / scale))

                _mean = np.array(bl_diffs)[idxs].mean()
                fp.write("| AVG BONE LENGTH DIFF               {:>10.4f}\n".format(_mean / scale))

                _mean = np.array(blratio_diffs)[idxs].mean()
                fp.write("| AVG BONE LENGTH RATIO DIFF         {:>10.4f}\n".format(_mean))

                _mean = np.array(vels)[idxs].mean()
                fp.write("| AVG COORDINATE VARIANCE            {:>10.4f}\n".format(_mean / scale))

                fp.write("-" * 55 + "\n\n")

        fp.write("-" * 55 + "\n")
        fp.write("| ALL CHARACTERS.                      {:^10}\n".format(" MEAN"))
        fp.write("-" * 55 + "\n")
        _mean = np.mean(avgs)
        fp.write("|                                    {:>10.4f}\n".format(_mean))
        fp.write("-" * 55 + "\n\n")
        avgs.append(_mean)

        fp.write("-" * 55 + "\n")
        for avg in avgs:
            fp.write("{},".format(avg))
        fp.write("\n" + "-" * 55 + "\n\n")
