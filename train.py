import argparse
import importlib
import os

import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter

from core.saver import Saver

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--config',
                        help='configure file name of the experiment',
                        default='ours.adv_quat_direct.full_sep',
                        type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    user_config = importlib.import_module('user_config.' + args.config)

    configG = user_config.GenConfig()
    configD = user_config.DiscConfig()
    if not isinstance(configG.dataset, list):
        configG.dataset = [configG.dataset]

    torch.set_default_dtype(configG.dtype)

    print("set config G: %s" % configG)
    print("set config D: %s" % configD)

    saverG = Saver(configG)
    modelG, optimG, global_step, last_epoch = saverG.load()

    saverD = Saver(configD)
    modelD, optimD, _, _ = saverD.load()

    models = EasyDict({
        'G': modelG,
        'D': modelD
    })

    optims = EasyDict({
        'G': optimG,
        'D': optimD
    })

    save_modelG = modelG
    save_modelD = modelD

    if configG.multi_gpu:
        models.G = torch.nn.DataParallel(modelG)
        models.D = torch.nn.DataParallel(modelD)

    schedulerG, schedulerD = None, None
    if hasattr(configG, 'scheduler'):
        configG.scheduler_param['last_epoch'] = -1
        schedulerG = configG.scheduler(optims.G, **configG.scheduler_param)
        schedulerD = configG.scheduler(optims.D, **configG.scheduler_param)

    log_dir = os.path.join(configG.ckpt_path, 'tb', configG.start_time)
    writer = SummaryWriter(log_dir=log_dir)
    trainer = configG.trainer(configG, models, optims, writer)

    for epoch in range(last_epoch + 1, configG.epoch):
        _, global_step, avg_loss = trainer.step(epoch, global_step)
        print('Training epoch %d was done. (avg_loss: %f)' % (epoch, avg_loss))

        print('Saving the trained generator model... (%d epoch, %d step)' % (epoch, global_step))
        saverG.save(save_modelG, optimG, global_step, epoch)
        print('Saving G is finished.')

        print('Saving the trained discriminator model... (%d epoch, %d step)' % (epoch, global_step))
        saverD.save(save_modelD, optimD, global_step, epoch)
        print('Saving D is finished.')

        if schedulerG:
            schedulerG.step(epoch)
            schedulerD.step(epoch)


if __name__ == '__main__':
    main()
