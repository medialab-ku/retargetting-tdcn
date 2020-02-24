import importlib
from torch.utils.data import ConcatDataset


def concat_dataset(dataset, dataset_param):
    dataset_list = []
    for ds in dataset:
        param = {
            **dataset_param,
        }
        dataset_list.append(ds(**param))

    if len(dataset_list) > 1:
        return ConcatDataset(dataset_list)
    else:
        return dataset_list[0]
