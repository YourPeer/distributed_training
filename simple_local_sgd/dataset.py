import random
import torch
from torchvision import datasets, transforms
import torch.distributed as dist
import numpy as np
""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

""" Partitioning MNIST """
def partition_dataset(batch_size,computing_power_difference):
    dataset = datasets.CIFAR10('../../data', train=True, download=False,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ]))

    size = dist.get_world_size()
    bsz = int(batch_size / float(size))
    # partition_sizes = [1.0 / size for _ in range(size)]
    partition_sizes = computing_power_difference/np.sum(computing_power_difference)
    # print(partition_sizes)
    partition = DataPartitioner(dataset, partition_sizes)

    partition = partition.use(dist.get_rank())
    print(partition.__len__())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True,
                                            drop_last=True)
    # for i,_ in enumerate(train_set):
    #     print(dist.get_rank(),i)

    testset = datasets.CIFAR10('../../data', train=False, download=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ]))
    test_set = torch.utils.data.DataLoader(testset,
                                        batch_size=bsz,
                                        shuffle=True,
                                           drop_last=True)
    return train_set, test_set, bsz