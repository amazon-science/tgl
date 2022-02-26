import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if label is not None:
            loss = self.loss_fn(x, label)
            return loss
        else:
            return x

batch_size = 2048
n_epochs = 20

train_data = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=16, pin_memory=True, prefetch_factor=20, persistent_workers=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=16, pin_memory=True, prefetch_factor=20, persistent_workers=True)

model = Net()
device = torch.device('cuda')
if torch.cuda.device_count() > 1:
    print('Using {} GPUs'.format(str(torch.cuda.device_count())))
    model = nn.DataParallel(model)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
tot_time = 0
for epoch in range(n_epochs):
    train_loss = 0.0
    model.train() # prep model for training
    t_tr = 0
    for data, target in train_loader:
        t_s = time.time()
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        loss = model(data, target)
        loss = loss.mean()
        loss.backward()
        import pdb; pdb.set_trace()
        optimizer.step()
        torch.cuda.synchronize()
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
    print('Epoch: {} \tTraining Loss: {:.6f}\tTest Acc: {:.2f}\tTime:{:.2f}'.format(epoch+1, train_loss, 100. * npred / ntot, t_tr))
print('Total Time: {:.2f}'.format(tot_time))