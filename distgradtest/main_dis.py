import os
import argparse
import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torchvision import datasets
from torch.utils.data.distributed import DistributedSampler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Permute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, label, total_bs, device):
        # all gather x and label
        x_global = [torch.zeros(i, x.shape[1], device=device) for i in total_bs]
        torch.distributed.all_gather(x_global, x)
        x_global = torch.cat(x_global, dim=0)
        label_global = [torch.zeros(i, device=device, dtype=label.dtype) for i in total_bs]
        torch.distributed.all_gather(label_global, label)
        label_global = torch.cat(label_global, dim=0)
        # generate a permutation
        if args.local_rank == 0:
            perm = torch.randperm(torch.sum(total_bs), device=device)
        else:
            perm = torch.randperm(torch.sum(total_bs), device=device)
        torch.distributed.broadcast(perm, 0)
        # get local shuffled x
        total_bs_cum = torch.cumsum(total_bs, dim=0)
        if args.local_rank == 0:
            x = x_global[perm][:total_bs_cum[0]]
            label = label_global[perm][:total_bs_cum[0]]
        else:
            x = x_global[perm][total_bs_cum[args.local_rank - 1]:total_bs_cum[args.local_rank]]
            label = label_global[perm][total_bs_cum[args.local_rank - 1]:total_bs_cum[args.local_rank]]
        # save for backward
        ctx.bs = total_bs
        ctx.perm = perm
        return x, label

    @staticmethod
    def backward(ctx, x_grad, label_grad):
        # all gather x_grad
        x_grad_global = [torch.zeros(i, x_grad.shape[1], device=device) for i in ctx.bs]
        torch.distributed.all_gather(x_grad_global, x_grad)
        x_grad_global = torch.cat(x_grad_global, dim=0)
        # inverse the permutaion
        perm_inv = torch.argsort(ctx.perm)
        # get true x_grad
        total_bs_cum = torch.cumsum(ctx.bs, dim=0)
        if args.local_rank == 0:
            x_gard = x_grad_global[perm_inv][:total_bs_cum[0]]
        else:
            x_grad = x_grad_global[perm_inv][total_bs_cum[args.local_rank - 1]:total_bs_cum[args.local_rank]]
        return torch.zeros_like(x_grad), None, None, None

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, label=None, permute=False, total_bs=None, device=None):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        if permute:
            x, label = Permute.apply(x, label, total_bs, device)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if label is not None:
            loss = self.loss_fn(x, label)
            return loss
        else:
            return x

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

# set which GPU to use
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)

# global group
torch.distributed.init_process_group(backend='nccl')
cpu_group = torch.distributed.new_group(backend='gloo')
set_seed(0)

batch_size = 2048 // 4
n_epochs = 20

train_data = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=20, persistent_workers=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=20, persistent_workers=True)

train_sampler = DistributedSampler(train_data)
train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
test_sampler = DistributedSampler(test_data)
test_loader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model = Net()
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
tot_time = 0
for epoch in range(n_epochs):
    train_loss = 0.0
    model.train() # prep model for training
    t_tr = 0
    for data, target in train_loader:
        # get batch size in all trainers
        local_bs = torch.zeros(torch.distributed.get_world_size(), dtype=torch.int)
        local_bs[args.local_rank] = data.shape[0]
        torch.distributed.all_reduce(local_bs, group=cpu_group)
        local_bs = local_bs.to(device)
        t_s = time.time()
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        loss = model(data, target, permute=True, total_bs=local_bs, device=device)
        loss = loss.mean()
        loss.backward()
        # if args.local_rank == 0:
        #     print('before: ', model.module.fc1.weight[0][:5])
        optimizer.step()
        torch.cuda.synchronize()
        # if args.local_rank == 0:
        #     print('after: ', model.module.fc1.weight[0][:5])
        t_tr += time.time() - t_s
        train_loss += loss.item()*data.size(0)
    if epoch > 2:
        tot_time += t_tr
    train_loss = train_loss/len(train_loader.dataset)
    test_loss = 0.0
    npred = 0
    ntot = 0
    model.eval() # prep model for *evaluation*
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        ntot += correct.shape[0]
        npred += int(correct.sum())
    if args.local_rank == 0:
        print('Epoch: {} \tTraining Loss: {:.6f}\tTest Acc: {:.2f}\tTime:{:.2f}'.format(epoch+1, train_loss, 100. * npred / ntot, t_tr))
if args.local_rank == 0:
    print('Total Time: {:.2f}'.format(tot_time))