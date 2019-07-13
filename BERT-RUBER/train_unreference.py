#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.7.10

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import argparse
import random
import os
import sys
import time
import pprint
import csv
import math
from tqdm import tqdm

from unreference_score import *
from utils import *

# set the random seed for the model
random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
    
    
def train(data_iter, net, optimizer, grad_clip=10):
    net.train()
    batch_num, losses = 0, 0
    criterion = nn.BCELoss()
    
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, label = batch
        qbatch = torch.from_numpy(qbatch)
        rbatch = torch.from_numpy(rbatch)
        label = torch.from_numpy(label).float()
        batch_size = qbatch.shape[0]
        
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            label = label.cuda()
            
        optimizer.zero_grad()
        
        scores = net(qbatch, rbatch)
        loss = criterion(scores, label)
        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        
        losses += loss.item()
        batch_num = batch_idx + 1
    return round(losses / batch_num, 4)

def validation(data_iter, net):
    net.eval()
    losses, batch_num, acc, acc_num = 0, 0, 0, 0
    criterion = nn.BCELoss()
    
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, label = batch 
        qbatch = torch.from_numpy(qbatch)
        rbatch = torch.from_numpy(rbatch)
        label = torch.from_numpy(label).float()
        batch_size = qbatch.shape[0]
                
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            label = label.cuda()
            
        scores = net(qbatch, rbatch)
        loss = criterion(scores, label)
        
        s = scores >= 0.5
        acc += torch.sum(s.float() == label).item()
        acc_num += batch_size
        
        batch_num += 1
        losses += loss.item()
        
    return round(losses / batch_num, 4), round(acc / acc_num, 4)


def test(net, data_iter):
    test_loss, test_acc = validation(data_iter, net)
    
    print('[!] test_loss:', test_loss)
    print('[!] test_Acc', test_acc)
    
    
def main(trainqpath, trainrpath, devqpath, devrpath, testqpath, testrpath, 
         weight_decay=1e-4, lr=1e-3):
    net = BERT_RUBER_unrefer(768, dropout=0.5)
    if torch.cuda.is_available():
        net.cuda()
    print('[!] Finish init the model')
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    epoches, grad_clip = 100, 10
    pbar = tqdm(range(1, epoches + 1))
    training_losses, validation_losses, validation_metrices = [], [], []
    min_loss = np.inf
    best_metric = -1
    
    os.system(f'rm ./ckpt/*')
    print(f'[!] Clear the checkpoints under ckpt')
    
    patience = 0
    begin_time = time.time()
    
    for epoch in pbar:
        train_iter = get_batch(trainqpath, trainrpath, 256)
        dev_iter = get_batch(devqpath, devrpath, 256)
        test_iter = get_batch(testqpath, testrpath, 256)
        
        training_loss = train(train_iter, net, optimizer)
        validation_loss, validation_metric = validation(dev_iter, net)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        validation_metrices.append(validation_metric)
        
        if best_metric < validation_metric:
            patience = 0
            best_metric = validation_metric
            min_loss = validation_loss
        else:
            patience += 1
            
        state = {'net': net.state_dict(), 
                 'optimizer': optimizer.state_dict(), 
                 'epoch': epoch}
        torch.save(state,
            f'./ckpt/Acc_{validation_metric}_vloss_{validation_loss}_epoch_{epoch}.pt')
        
        pbar.set_description(f"loss(train-dev): {training_loss}-{validation_loss}, Acc: {validation_metric}, patience: {patience}")
    pbar.close()
    
    # calculate costing time
    end_time = time.time()
    hour = math.floor((end_time - begin_time) / 3600)
    minute = math.floor(((end_time - begin_time) - 3600 * hour) / 60)
    second = (end_time - begin_time) - hour * 3600 - minute * 60
    print(f"Cost {hour}h, {minute}m, {round(second, 2)}s")
    
    # load best and test
    load_best_model(net)
    
    # test
    test(net, test_iter)
    


if __name__ == "__main__":
    main('data/src-train.embed',
         'data/tgt-train.embed',
         'data/src-dev.embed',
         'data/tgt-dev.embed',
         'data/src-test.embed',
         'data/tgt-test.embed')
