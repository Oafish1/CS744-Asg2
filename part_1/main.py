import argparse
import copy
import json
import logging
import os
import random
from time import perf_counter

import model as mdl
import numpy as np
import torch
import torch.distributed as dd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Parse arguments (But don't use them)
parser = argparse.ArgumentParser()
parser.add_argument('--master-ip', dest='master_ip', type=str)
parser.add_argument('--master-port', dest='master_port', default=6585, type=int)
parser.add_argument('--num-nodes', dest='num_nodes', type=int)
parser.add_argument('--rank', dest='rank', type=int)
args = parser.parse_args()
MASTER_IP = args.master_ip
MASTER_PORT = args.master_port
NUM_NODES = args.num_nodes
RANK = args.rank

# Seeding
np.random.seed(42)
torch.manual_seed(42)

# Setup
device = 'cpu'
torch.set_num_threads(4)
BATCH_SIZE = 256 # batch for one node
CLI_INTERVAL = 20 # log every _ batches
SAMPLE_INTERVAL = (1, 40) # Not right-side inclusive


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Add timer
        if batch_idx == SAMPLE_INTERVAL[0]:
            start_time = perf_counter()
        elif batch_idx == SAMPLE_INTERVAL[1]:
            duration = perf_counter() - start_time
            avg_time = duration / (SAMPLE_INTERVAL[1] - SAMPLE_INTERVAL[0])
            print(f'Average batch times {SAMPLE_INTERVAL}: {avg_time}')

        # Make data usable
        data, target = data.to(device), target.to(device)

        # Forward
        logits = model(data)

        # Backward (Compute Gradient)
        optimizer.zero_grad()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        # Log every `CLI_INTERVAL` batches
        if batch_idx % CLI_INTERVAL == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Local loss: {loss}')

    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=BATCH_SIZE,
                                                    sampler=None,
                                                    shuffle=True,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    main()
