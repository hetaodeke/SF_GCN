import argparse
from ast import parse
from feeder.skeleton_helper import random_move
import os
import random
import shutil
import time
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

from model.SF_GCN import SF_GCN
from feeder.skeleton_dataset import Skeleton
from utils import *


parser = argparse.ArgumentParser(description="Pytorch SF_GCN Training")
parser.add_argument('--gpu_idx', default=[0,1,2,3], type=list, help='list of used gpus index')
parser.add_argument('--base_lr', default=0.1, type=float, help='base learning_rate')
parser.add_argument('--batch_size', default=64, type=int, help='number of batch size')
parser.add_argument('--epoch', default=100, type=int, help='number of training epoch')

parser.add_argument('--train_data_path', default='data/NTU-RGB-D/xsub/train_data.npy', type=str ,help='path of training original data file')
parser.add_argument('--train_label_path', default='data/NTU-RGB-D/xsub/train_label.pkl', type=str ,help='path of training original label file')
parser.add_argument('--val_data_path', default='data/NTU-RGB-D/xsub/val_data.npy', type=str ,help='path of valing original data file')
parser.add_argument('--val_label_path', default='data/NTU-RGB-D/xsub/val_label.pkl', type=str ,help='path of valing original label file')

parser.add_argument('--alpha', default=8, type=int, help='alpha times temporal resolution')
parser.add_argument('--beta_inv', default=8, type=int, help='beta times channel dimension')

parser.add_argument('--save_model', default='checkpoints/best_model.pth', type=str, help='model save path')
parser.add_argument('--log_dir', default='log/{}.log'.format(time.strftime(r"%Y-%m-%d-%H_%M_%S", time.localtime())), type=str, help='path of log file')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=int(time.time() * 256), type=int, help='seed for initializing training. ')
args = parser.parse_args()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main(args):
    best_acc = 0.
    
    train_dataset = Skeleton(args,
                            args.train_data_path, 
                            args.train_label_path,
                            random_choose=True,
                            random_move=True,
                            window_size=150,
                            debug=True
                            )
    val_dataset = Skeleton(args,
                            args.val_data_path, 
                            args.val_label_path,
                            random_choose=True,
                            random_move=True,
                            window_size=150,
                            debug=True
                            )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size * len(args.gpu_idx), shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size * len(args.gpu_idx), shuffle=True)
    
    # initialize model
    model = SF_GCN(
                    num_pathways=2,
                    in_channels=3,
                    max_frame=150,
                    num_class=80,
                    graph_args={'layout':'ntu-rgb+d', 'strategy':'spatial'},
                    beta_inv=args.beta_inv
                    )
    model_name = model._get_name()
    model.apply(weights_init)
    model = nn.DataParallel(model, device_ids=args.gpu_idx)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-7)
    criterion = nn.CrossEntropyLoss()

    # get logger 
    logger = get_logger(args)
    logger.info('Parameter:{ Batch_size:' + str(args.batch_size) +
                '\tbase_lr:' + str(args.base_lr) +  
                '\tModel:' + model_name + '}')

    # tesorboard writer
    writer = SummaryWriter()

    for epoch in range(args.epoch):
        print('Train Phase:')
        train(args, epoch, train_dataloader, model, criterion, optimizer, logger, writer)
        print('Val Phase:')
        acc1, acc5 = validation(model, val_dataloader, criterion, logger)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc
        best_acc = max(acc1, best_acc)

        if is_best:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc,
                }, args.save_model
                )

    print("Train finished! to see train infomation in log/, to see train result in runs/ by using tensorboard command")

def train(args, epoch, train_dataloader, model, criterion, optimizer, logger, writer):
    model.train()

    for iter, traindata in enumerate(train_dataloader):
        start = time.time()
        data, label = traindata
        if isinstance(data, (list,)):
            for i in range(len(data)):
                data[i] = data[i].cuda()
        else:
            data = data.cuda()
        label.cuda()

        output = model(data)
        loss = criterion(output, label)
        adjust_lr(optimizer, epoch, args.base_lr)

        acc1 = accuracytop1(output, label, (1, ))
        acc5 = accuracytop5(output, label, (5, ))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        if iter % args.print_freq == 0:
            logger.info('epoch:{} [{}/{}] loss:{:.4f} | acc@1:{:.4f} | acc@5:{:.4f} | time:{:.4f}s'.format(epoch, iter, len(train_dataloader), loss, acc1, acc5, end-start))
        
        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('Acc@1', acc1, epoch)
        writer.add_scalar('Acc@5', acc5, epoch)
        writer.flush()
        writer.close()

def validation(model, val_dataloader, criterion, logger):
    model.eval()
    
    with torch.no_grad():
        for iter, valdata in enumerate(val_dataloader):
            data, label = valdata
            if isinstance(data, (list,)):
                for i in range(len(data)):
                    data[i] = data[i].cuda()
            else:
                data = data.cuda()
            label.cuda()
            output = model(data)
            loss = criterion(output, label)
            acc1 = accuracytop1(output, label, (1, ))
            acc5 = accuracytop5(output, label, (5, ))

            if iter % args.print_freq == 0:
                logger.info('[{}/{}] | loss:{:.4f} | acc@1:{:.4f} | acc@5:{:.4f}'.format(iter, len(val_dataloader), loss, acc1, acc5))

if __name__ == '__main__':
    main(args)