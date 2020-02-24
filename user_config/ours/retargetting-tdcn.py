import os

import torch
import numpy as np

from core.config import Config

_binary_path = os.path.join("", "datasets", "mixamoquatdirect_train_large.bin")

_optim = torch.optim.Adam
_optim_param = {
    'lr': 0.0001,
    'betas': (0.5, 0.999),
}
_multi_gpu = False

from model.ours.retargetting_tdcn import RetargettingTDCN
from model.ours.trainer import OursTrainer
from model.ours.mixamo import MIXAMOdataset
from model.ours.discriminator import Discriminator
from torch.optim.lr_scheduler import MultiStepLR

model_param = {
    'n_joints': 22,
    ##
    'filter_widths': [3, 3, 3, 3],
    'dropout': 0.2,
    'channels': [1024, 1024, 1024, 1024],
    'out_features': 512,
    ##
    'causal': False,
    'dense': False,
    'parents': [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20],
    'binary_path': _binary_path
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GenConfig(Config):
    def __init__(self):
        super().__init__(
            name=__name__,
            tag='gen',
            model=RetargettingTDCN,
            model_param=model_param,
            dataset=[
                MIXAMOdataset
            ],
            dataset_param={
                'binary_path': _binary_path,
                'out_frame': 81,
                'scale_augment': True,
                'scale_std': 0.2,
                'dtype': np.float32,
                'num_critics': 1
            },
            trainer=OursTrainer,
            binomial_decay=False,
            n_batch=64,
            n_worker=8,
            device=device,
            loss=None,
            optim=_optim,
            optim_param=_optim_param,
            epoch=800,
            multi_gpu=_multi_gpu,

            scheduler=MultiStepLR,
            scheduler_param={
                'milestones': [100, 300],
                'gamma': 0.1,
            },

            # current settings
            dtype=torch.float32,
            n_joints=22,
            angle=100.0,
            localW=1.,
            quatW=10.,
            rtjW=5.,  # lambda_g
            autoW=1.,  # autoencoder loss in the adversarial loss (paper)
            cycleW=1.,
            twistW=10.,  # lambda_t
            advW=1.,  # lambda_a
            heightW=10.,  # lambda_h
            heightF=1.,
            heightAxis=[0, 1, 2],  # xyz
            ganAxis=[0, 1, 2],
            raquatW=0.1,  # lambda_r
            advScore=0.3,
            orientationW=1.,  # lambda_o

            discLambda=5.,
            endeffectors=[5, 17, 21, 9, 13]
        )


class DiscConfig(Config):
    def __init__(self):
        n_joints = 22
        super().__init__(
            name=__name__,
            tag='disc',
            model=Discriminator,
            model_param={
                'in_features': 5 + 1 + 22,
                'filter_widths': [3, 3, 3],
                'channels': 256,
                'leak': 0.2
            },
            dataset=None,
            dataset_param=None,
            trainer=None,
            validator=None,
            n_batch=None,
            n_worker=None,
            device=device,
            loss=None,
            optim=_optim,
            optim_param=_optim_param,
            epoch=None,
            multi_gpu=_multi_gpu,
        )


class EvalConfig(Config):
    def __init__(self):
        super().__init__(
            name=__name__,
            tag='eval',
            model=RetargettingTDCN,
            model_param=model_param,
            dataset=None,
            dataset_param=None,
            trainer=None,
            validator=None,
            n_batch=None,
            n_worker=None,
            device=device,
            loss=None,
            optim=torch.optim.Adam,
            optim_param=_optim_param,
            epoch=None,
            multi_gpu=False,

        )
