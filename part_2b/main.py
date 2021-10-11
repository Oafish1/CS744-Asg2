import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
from torch import distributed as dist
import sys
import argparse
import numpy
import time
device = "cpu"
torch.set_num_threads(4)

batch_size = 256 # batch for one node

parser = argparse.ArgumentParser()
parser.add_argument("--master-ip", help = "Maters's ip")
parser.add_argument("--num-nodes", type = int, help = "Node's num")
parser.add_argument("--rank", type = int, help = "rank")



def dist_setup(rank, world_size, master):
    dist.init_process_group(backend = "gloo",
                            init_method = master,
                            rank = rank,
                            world_size = world_size)



        

def train_model(model, train_loader, optimizer, criterion, epoch, world_size, rank):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    running_loss = 0.0
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your code goes here!
        

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        
        
        # Traverse all the layers
        for parameter in model.parameters():
            
            grad = parameter.grad
            # Doing the all_reduce
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad /= world_size
            
            parameter.grad = grad
        
        
             
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 20 == 19:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, batch_idx + 1, running_loss / 20))
            running_loss = 0.0

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
    # Parse the argument 
    args = parser.parse_args()
    world_size = args.num_nodes
    rank = args.rank
    master = args.master_ip

    dist_setup(rank, world_size, master)

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
    # Parttion the datae
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_set,
                num_replicas = world_size,
                rank = rank
            )

    # Setup the random seed
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(2048)
     
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=world_size,
                                                    batch_size=int(batch_size/4),
                                                    sampler=train_sampler,
                                                    worker_init_fn=seed_worker,
                                                    generator=g,
                                                    shuffle=(train_sampler is None),
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)
    

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)

    start = time.time()
    # running training for one epoch
    for epoch in range(1):
        train_sampler.set_epoch(epoch)
        train_model(model, train_loader, optimizer, training_criterion, epoch, world_size, rank)
        test_model(model, test_loader, training_criterion)
    
    end = time.time()
    print("time :" + str(end - start))


if __name__ == "__main__":
    main()
