import os
import shutil
import pprint
from abc import ABC, abstractmethod
from datetime import datetime


class Config(ABC):
    important_keys = [
        'name',
        'tag',
        'model',
        'model_param',
        'dataset',
        'dataset_param',
        'trainer',
        'n_batch',
        'n_worker',
        'device',
        'loss',
        'optim',
        'optim_param',
        'epoch',
    ]

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.multi_gpu = False
        self.max_to_keep = 50
        self.start_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.__dict__.update(kwargs)
        self._check_attr()

        self.cfg_file = os.path.join(*self.name.split('.')) + '.py'
        self.ckpt_path = os.path.join('ckpt', *self.name.split('.')[1:], self.tag)

        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        i = 0
        dst = os.path.join(self.ckpt_path, self.tag + '_({}).py'.format(i))
        exist = os.path.exists(dst)
        while exist:
            i += 1
            dst = os.path.join(self.ckpt_path, self.tag + '_({}).py'.format(i))
            exist = os.path.exists(dst)

        shutil.copy(self.cfg_file, dst)

    def _check_attr(self):
        for k in self.important_keys:
            if k not in self.__dict__.keys():
                raise KeyError("Config should have the key ({})".format(k))

    def __str__(self):
        return str(self.__class__) + ": \n" + pprint.pformat(self.__dict__, indent=4)
