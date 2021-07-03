import os
import sys
import logging
import builtins

import torch
import torch.distributed as dist


def accuracytop1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res[0]

def accuracytop5(output, target, topk=(5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res[0]

def adjust_lr(optimizer, epoch ,lr):
    """
    Sets the learning rate to the initial LR decayed by 10 every 20 epochs
    """
    lr = lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass

def is_master_proc(args):
    if torch.distributed.is_initialized():
        return dist.get_rank() % args.num_gpus == 0
    else:
        return True

def get_logger(args):
    logger = logging.getLogger(__name__)

    logger.setLevel(level=logging.INFO)
    logger.propagate = False
    if not os.path.exists(args.log_dir):
        f = open(args.log_dir, 'w')
        f.close()
    if is_master_proc(args):
        handler = logging.FileHandler(args.log_dir, encoding='utf-8')
        handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(level=logging.INFO)

        logger.addHandler(console)
        logger.addHandler(handler)
    else:
        _suppress_print()

    return logger
