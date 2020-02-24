from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from . import concat_dataset


class Trainer(ABC):

    @abstractmethod
    def __init__(self, config, model, optim, writer):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.optim = optim

        self.data_loader = DataLoader(
            concat_dataset(self.config.dataset, self.config.dataset_param),
            batch_size=self.config.n_batch,
            num_workers=self.config.n_worker,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        self.writer = writer

    @abstractmethod
    def step(self, epoch, step):
        pass
