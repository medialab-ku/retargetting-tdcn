import argparse
import importlib
import os
import pickle

import numpy as np
import torch
from easydict import EasyDict
from tqdm import trange

import core.anim.bvh as BVH
from core.quaternions import Quaternions
from core.saver import Saver
from utils.utils import copy_model, get_skel_height_np, get_bonelengths, compare_bls, create_report, \
    load_testdata_root, put_in_world, put_in_world_batch


def test_mixamo(model, path, device, max_steps=None, testset='nkn'):
    suffix = ''
    if max_steps:
        suffix += "max{}frame".format(max_steps)
    else:
        suffix += "fullseq"
    suffix += testset

    min_steps = 120
    if testset == 'paper':
        min_steps = 0

    with open('./datasets/mixamoquatdirect_train_large.bin', 'rb') as binary:
        b = pickle.load(binary)

    f_np = EasyDict(b['f'])
    local_mean = f_np.local.mean
    local_std = f_np.local.std
    rtj_mean = f_np.rtj.mean
    rtj_std = f_np.rtj.std

    f = EasyDict({
        'local': {
            'mean': torch.from_numpy(local_mean).float().to(device),
            'std': torch.from_numpy(local_std).float().to(device),
        },
        'rtj': {
            'mean': torch.from_numpy(rtj_mean).float().to(device),
            'std': torch.from_numpy(rtj_std).float().to(device),
        },
    })
    diff_worlds, diff_locals, diff_rtjs, vels, \
    labels, bl_diffs, blratio_diffs \
        = list(), list(), list(), list(), list(), list(), list()

    (inpjoints, inpanims, inpquats, inplocals, inprtjs, inpskels, from_names,
     tgtjoints, tgtanims, tgtquats, tgtlocals, tgtrtjs, tgtskels, to_names,
     gtanims) = \
        load_testdata_root(min_steps, max_steps, b['scale'], testset)

    gt_world_pos, keys = list(), list()
    inp_heights = get_skel_height_np(np.asarray(inpskels))
    gt_heights = get_skel_height_np(np.asarray(tgtskels))
    for i in range(len(tgtlocals)):
        gt = put_in_world(tgtlocals[i].copy(),
                          tgtrtjs[i].copy(),
                          tgtanims[i][0].rotations[:, 0])

        km = from_names[i].split("_")[0:2]
        from_ = from_names[i][:sum([len(l) for l in km]) + 1]
        kc = to_names[i].split("_")[0:2]
        to_ = to_names[i][:sum([len(l) for l in kc]) + 1]
        key = from_ + "/" + to_

        gt_world_pos.append(gt)
        keys.append(key)

    n_inpskels, n_tgtskels = list(), list()
    for i in range(len(inplocals)):
        inplocals[i] = (inplocals[i] - local_mean) / local_std
        inprtjs[i] = (inprtjs[i] - rtj_mean) / rtj_std
        n_inpskels.append((inpskels[i] - local_mean) / local_std)
        n_tgtskels.append((tgtskels[i] - local_mean) / local_std)

    results_dir = os.path.join(path, "test_{}".format(suffix))

    with trange(len(inplocals)) as tq:
        for i in tq:
            tq.set_description('Mixamo Testing')
            steps = inplocals[i].shape[0]
            tq.set_postfix(frame=steps)

            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            quatA = torch.tensor(inpquats[i], dtype=torch.float32, device=device).view([1, steps, 22, 4])
            rtjA = torch.tensor(inprtjs[i], dtype=torch.float32, device=device).view(1, steps, 3)
            skelA = np.asarray(n_inpskels[i])[0:1]
            skelA = torch.tensor(skelA, dtype=torch.float32, device=device).view(1, 1, 22, 3)
            skelB = np.asarray(n_tgtskels[i])[0:1]
            skelB = torch.tensor(skelB, dtype=torch.float32, device=device).view(1, 1, 22, 3)

            dlocalB = torch.tensor(tgtlocals[i], dtype=torch.float32, device=device)
            drtjB = torch.tensor(tgtrtjs[i], dtype=torch.float32, device=device).view(1, steps, 3)

            with torch.no_grad():
                _, (clocalB, cdlocalB, crtjB, cquatB, _) = model(quatA, rtjA, skelA, skelB)
                cdrtjB = crtjB * f.rtj.std + f.rtj.mean

                init_diff = drtjB[:, 0:1] - cdrtjB[:, 0:1]
                cdrtjB = cdrtjB + init_diff

                diff_local = ((cdlocalB - dlocalB) ** 2).sum(dim=-1).cpu().numpy()[0]
                diff_local = 1. / gt_heights[i] * diff_local

                diff_rtj = (cdrtjB - drtjB) ** 2
                diff_rtj = diff_rtj.sum(dim=-1).cpu().numpy()[0]

                outputB = put_in_world_batch(cdlocalB, cdrtjB, cquatB[:, :, 0]).cpu().numpy()[0]
                diff_world = 1. / gt_heights[i] * ((outputB - gt_world_pos[i]) ** 2).sum(axis=-1)

                variance = np.std(gt_world_pos[i], axis=0) ** 2
                variance = 1. / gt_heights[i] * variance

                tgtbls = get_bonelengths(tgtskels[i])
                inpbls = get_bonelengths(inpskels[i])
                bl_diff, blratio_diff = compare_bls(tgtbls, inpbls)

                labels.append(keys[i])
                diff_worlds.append(diff_world)
                diff_locals.append(diff_local)
                diff_rtjs.append(diff_rtj)
                vels.append(variance)
                bl_diffs.append(bl_diff)
                blratio_diffs.append(blratio_diff)

                cdlocalB = cdlocalB.cpu().numpy()[0]
                cdrtjB = cdrtjB.cpu().numpy()[0]
                cquatB = cquatB.cpu().numpy()[0]

            # create bvh path
            if "Big_Vegas" in from_names[i]:
                from_bvh = from_names[i].replace("Big_Vegas", "Vegas")
            else:
                from_bvh = from_names[i]

            if "Warrok_W_Kurniawan" in to_names[i]:
                to_bvh = to_names[i].replace("Warrok_W_Kurniawan", "Warrok")
            else:
                to_bvh = to_names[i]

            bvh_path = os.path.join(path, 'blender_files_{}'.format(suffix), to_bvh.split("_")[-1])
            if not os.path.exists(bvh_path):
                os.makedirs(bvh_path)
            bvh_path += "/{0:05d}".format(i)

            # save input sequence
            inpanim, inpnames, inpftime = inpanims[i]
            BVH.save(bvh_path + "_from=" + from_bvh + "_to=" + to_bvh + "_inp.bvh", inpanim, inpnames, inpftime)

            # save ground truth
            gtanim, gtnames, gtftime = gtanims[i]
            BVH.save(bvh_path + "_from=" + from_bvh + "_to=" + to_bvh + "_gt.bvh", gtanim, gtnames, gtftime)

            # save the retargeted result
            tgtanim, tgtnames, tgtftime = tgtanims[i]

            """Exclude angles in exclude_list as they will rotate non-existent
               children During training."""
            exclude_list = [5, 17, 21, 9, 13]
            anim_joints = []
            quat_joints = []
            for l in range(len(tgtjoints[i])):
                if l not in exclude_list:
                    anim_joints.append(tgtjoints[i][l])  # excluded joint index w.r.t all joints
                    quat_joints.append(l)  # excluded joint index

            root_rotB = Quaternions(cquatB[:, 0])

            worldB = put_in_world(cdlocalB, cdrtjB, root_rotB)

            tgtanim.positions[:, 0] = worldB[:, 0].copy()
            excluded_quatB = cquatB[:, quat_joints].copy()
            excluded_quatB[:, 0, :] = root_rotB.qs
            tgtanim.rotations.qs[:, anim_joints] = excluded_quatB
            BVH.save(bvh_path + "_from=" + from_bvh + "_to=" + to_bvh + ".bvh", tgtanim, tgtnames, tgtftime)

    if testset == 'nkn':
        print("Create report to {}...".format(results_dir))
        create_report(diff_worlds, diff_locals, diff_rtjs, vels,
                      labels, bl_diffs, blratio_diffs, inp_heights, gt_heights,
                      suffix, results_dir)
        print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description='Target configuration')
    parser.add_argument('--config',
                        help='The configuration file',
                        default='ours.retargetting-tdcn',
                        type=str)

    parser.add_argument('--epoch',
                        help='The target epoch of the configuration',
                        default='ours.retargetting-tdcn',
                        type=int)

    parser.add_argument('--shortseq',
                        help='Mixamo evalution for short sequence (120 frame)',
                        default=True,
                        action='store_true')
    parser.add_argument('--no-shortseq', dest='shortseq', action='store_false')

    parser.add_argument('--fullseq',
                        help='Mixamo evalution for full sequence (no limit frame)',
                        default=True,
                        action='store_true')
    parser.add_argument('--no-fullseq', dest='fullseq', action='store_false')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    user_config = importlib.import_module('user_config.' + args.config)
    config = user_config.EvalConfig()
    if not isinstance(config.dataset, list):
        config.dataset = [config.dataset]

    copy_model(args.config, args.epoch)

    saver = Saver(config)
    model, optim, global_step, epoch = saver.load()

    if config.multi_gpu:
        model = torch.nn.DataParallel(model,
                                      device_ids=[0, 1],
                                      output_device=0)

    model.eval()

    results_path = "./results/{}".format(args.config.replace('.', '_'))

    if args.shortseq:
        test_mixamo(model, results_path, config.device, max_steps=120)

    if args.fullseq:
        test_mixamo(model, results_path, config.device, max_steps=None)


if __name__ == "__main__":
    main()
