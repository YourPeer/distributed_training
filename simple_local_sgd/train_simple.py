from Net import Net,resnet18
import os
import torch.distributed as dist
from torch.multiprocessing import Process
import torch
from dataset import partition_dataset
import torch.optim as optim
from math import ceil
import torch.nn as nn
import time
import argparse
import pandas as pd
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,4'
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
    parser.add_argument('--size', default=4, help='total gpu num')
    parser.add_argument('--epoches', default=1, help='epoch num')
    parser.add_argument('--lr', default=0.01, help='learning rate')
    parser.add_argument('--tau', default=10, help='how much step to all_reduce')
    parser.add_argument('--batch_size', default=128, help='batch_size')
    parser.add_argument('--print_feq', default=30, help='print log per n step')
    parser.add_argument('--csv_record_file_name', default='lab1_record.csv', help='record data')
    args = parser.parse_args()
    return args

def init_process(rank, size, fn, args,backend='nccl'):
    """ Initialize the distributed environment. """

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size,args)

def average_gradients(model,step,rank,tau):
    if step%tau==0:
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        return True
    return False

def my_eval_net(model, test_set):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data,target in test_set:
            data=data.cuda(non_blocking = True)
            target=target.cuda(non_blocking = True)
            outputs=model(data)
            _, pre = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (pre == target).sum().item()
    model.train()
    return 100 * correct // total

def run(rank, size, args):
    torch.manual_seed(1234)
    computing_power_difference = [4, 1,4,1]
    train_set, test_set, bsz = partition_dataset(args.batch_size,computing_power_difference)
    torch.cuda.set_device(rank)
    model = resnet18().cuda(rank)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=0.5)
    criterion = nn.CrossEntropyLoss().cuda(rank)
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    loss_list=[]
    time_list=[]
    acc_list=[]
    total_train_time_list=[]
    global_step = 0
    dist.barrier()
    start_train = time.time()
    for epoch in range(args.epoches):
        epoch_loss = 0.0
        start = time.time()
        local_time=0

        for step,(data, target) in enumerate(train_set):

            data=data.cuda(rank,non_blocking=True)
            target=target.cuda(rank,non_blocking=True)

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            epoch_loss += loss.item()

            loss.backward()

            if rank==dist.get_rank() and local_time<computing_power_difference[dist.get_rank()]-1:
                #print('rank:',dist.get_rank(),'local_time:',local_time)
                optimizer.step()
                local_time+=1
                continue
            else:
                optimizer.step()
                local_time=0

            torch.cuda.synchronize()
            dist.barrier()
            global_step+=1
            is_average=average_gradients(model,global_step,rank,args.tau)

            if is_average:
                optimizer.step()
            if rank == 0 and is_average:
                print('average!!!')
            if rank==0 and global_step%args.print_feq==0:
                print('global_step:',global_step,',Rank:',rank,',loss:',loss.item())
            print("rank",dist.get_rank(),"step:",step)
        if rank==dist.get_rank():
            print('rank '+dist.get_rank()+' training over!')
        eval_accuracy=my_eval_net(model, test_set)
        over=time.time()
        use_time=over-start
        train_time = over-start_train
        loss_list.append(epoch_loss / num_batches)
        time_list.append(use_time)
        acc_list.append(eval_accuracy)
        total_train_time_list.append(train_time)
        if rank==0:
            print('Rank:', dist.get_rank(), ', epoch:',epoch, ': ', epoch_loss / num_batches,',use_time:',use_time)

    record_dict={"tau="+str(args.tau)+"  loss":loss_list,"    time":time_list,"    accuracy":acc_list,"    total_train_time":total_train_time_list}
    tag = 'lr{:.3f}_bs{:d}_tau{:d}_ws{}_epoch{:d}.csv'# 'bs' means batch_size, 'ws' means world_size
    saveFileName = tag.format(args.lr, args.batch_size, args.tau, args.size, args.epoches)

    csv_file=pd.DataFrame(record_dict)
    csv_file.to_csv(saveFileName)

# if __name__ == "__main__":
#     args = get_args()
#     processes = []
#     tau_list = [1, 10, 20, 40, 80, 100, 195, 390, 400]
#     for tau in tau_list:
#         args.tau = tau
#         for rank in range(args.size):
#             p = Process(target=init_process, args=(rank, args.size, run, args))
#             p.start()
#             processes.append(p)
#
#         for p in processes:
#             p.join()

if __name__ == "__main__":
    args = get_args()
    processes = []
    for rank in range(args.size):
        p = Process(target=init_process, args=(rank, args.size, run, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
