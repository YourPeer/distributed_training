#system packages
import torch
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist

#custom packages
from trian_tools import train_one_epoch,eval_one_epoch
from model import resnet18
from dataset import partition_dataset

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
    parser.add_argument('--data', default='../data', help='path to dataset')
    parser.add_argument('--checkpoint', default='../checkpoints/best_accuracy.pth', help='path to checkpoint')
    parser.add_argument('--world_size', default=2, help='total gpu num')
    parser.add_argument('--epoches', default=1, help='epoch num')
    parser.add_argument('--lr', default=0.01, help='learning rate')
    parser.add_argument('--tau', default=80, help='how much step to all_reduce')
    parser.add_argument('--batch_size', default=128, help='batch_size')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    args = parser.parse_args()
    return args

def train(rank,nprocs,args):
    print(rank)
    torch.distributed.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            rank=rank,
                            world_size=args.world_size)
    # seed for reproducibility
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    torch.backends.cudnn.deterministic = True
    # create dataset.
    train_loader, test_loader = partition_dataset(rank, args.size, args)
    print(train_loader)
    # create model.

    # define the optimizer.

    # define the lr scheduler.

def main():
    args =get_args() #??????
    print("The config parameters are-> world_size:%d, epoches:%d, lr:%.2f, tau:%d" % (
    args.world_size, args.epoches, args.lr, args.tau))
    import time
    start = time.time()
    mp.spawn(train, nprocs=args.world_size, args=(args.world_size, args))
    end = time.time()
    print("Training time is: " + str(end - start))

if __name__ == '__main__':
    main()