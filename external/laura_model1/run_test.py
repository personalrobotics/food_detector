#!/usr/bin/env python


from __future__ import print_function
from __future__ import division

import numpy as np
import sys
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # input_size = (-1, 32, 32, 3)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)

        x = F.dropout(x, p=0.4, training=self.training)
        x = x.view(-1, 64 * 8 * 8)

        x = self.fc_layers(x)
        return x


class Model1Dataset(data.Dataset):
    def __init__(self, base_dir, transform=None, train=True):
        self.base_dir = base_dir

        self.pos_dirname = os.path.join(self.base_dir, 'success')
        self.neg_dirname = os.path.join(self.base_dir, 'failure')

        pos_filenames = glob.glob(os.path.join(self.pos_dirname, '*.png'))
        neg_filenames = glob.glob(os.path.join(self.neg_dirname, '*.png'))
        self.img_filenames = pos_filenames + neg_filenames
        self.labels = np.hstack(
            (np.ones(len(pos_filenames)),
             np.zeros(len(neg_filenames))))

        self.num_samples = len(self.labels)
        self.transform = transform

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        label = self.labels[idx]

        img = Image.open(img_filename)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.Tensor(img)

        label = torch.FloatTensor([label])
        return img, label

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        return torch.stack(imgs), torch.stack(labels)

    def __len__(self):
        return self.num_samples


def train_model1():
    print('train_lwp_model1')

    checkpoint_path = './checkpoint/model1_ckpt.pth'
    best_loss = float('inf')
    start_epoch = 0

    transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor()])

    print('Loading dataset')
    trainset = Model1Dataset(
        base_dir='./data/train',
        transform=transform,
        train=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=16,
        shuffle=True, num_workers=8,
        collate_fn=trainset.collate_fn)

    testset = Model1Dataset(
        base_dir='./data/validate',
        transform=transform,
        train=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4,
        shuffle=True, num_workers=8,
        collate_fn=testset.collate_fn)

    net = Model1()
    print(net)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    net = net.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    print('trainset: {}'.format(trainloader.dataset.num_samples))
    print('testset: {}'.format(testloader.dataset.num_samples))

    for epoch in range(start_epoch, start_epoch + 1000):
        print('\nEpoch: {}'.format(epoch))

        # training
        net.train()
        train_loss = 0
        total_batches = int(np.ceil(
            trainloader.dataset.num_samples / trainloader.batch_size))

        for idx, item in enumerate(trainloader):
            img = item[0]
            gt_label = item[1]
            img = img.cuda()
            gt_label = gt_label.cuda()

            optimizer.zero_grad()
            pred_label = net(img)

            loss = criterion(pred_label, gt_label)
            loss.backward()
            optimizer.step()

            train_loss += loss.data

            if idx % 10 == 0:
                print('[{0}| {1:03d}/{2:03d}] train_loss: {3:.3f} | avg_loss: {4:.3f}'.format(
                    epoch, idx, total_batches, loss.data, train_loss / (idx + 1)))

        # testing
        print('\nTest')
        net.eval()
        test_loss = 0
        total_batches = int(np.ceil(
            testloader.dataset.num_samples / testloader.batch_size))

        for idx, item in enumerate(testloader):
            img = item[0]
            gt_label = item[1]
            img = img.cuda()
            gt_label = gt_label.cuda()

            pred_label = net(img)
            loss = criterion(pred_label, gt_label)

            test_loss += loss.data

            if idx % 10 == 0:
                print('[{0}| {1:03d}/{2:03d}]  test_loss: {3:.3f} | avg_loss: {4:.3f}'.format(
                    epoch, idx, total_batches, loss.data, test_loss / (idx + 1)))

        # save checkpoint
        test_loss /= len(testloader)
        if test_loss < best_loss:
            print('Saving checkpoint..')
            state = {
                'net': net.module.state_dict(),
                'loss': test_loss,
                'epoch': epoch
            }
            if not os.path.exists(os.path.dirname(checkpoint_path)):
                os.makedirs(os.path.dirname(checkpoint_path))
            torch.save(state, checkpoint_path)
            best_loss = test_loss


def test_model1():
    print('test_lwp_model1')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        train_model1()
    else:
        test_model1()
