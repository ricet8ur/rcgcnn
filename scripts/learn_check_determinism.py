# this script checks consistency of learning on the same data with the same random_key
import pandas as pd
import json
import numpy as np
from subprocess import run
import shutil as st
import argparse
import main
import sys
import torch
parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options',default=['./data/root/data/'], metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.6, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


fields = [
    "energy_per_atom",
    "formation_energy_per_atom",
    "band_gap",
    "efermi",
    "k_voigt",
    "k_reuss",
    "k_vrh",
    "g_voigt",
    "g_reuss",
    "g_vrh",
    "homogeneous_poisson",
]

def set_property_to_ids(df: pd.DataFrame, prop: str):
    part_df = df[prop].dropna()
    part_df = part_df.groupby(part_df.index).first()
    part_df.to_csv("./data/root/data/id_prop.csv", index=True, header=False)

non_ill_df = pd.read_csv("./non_ill_df.csv",index_col=0)
print(non_ill_df.info())
resdict = dict()
maedict = dict()

import time
a = time.time()
for k in range(1):
    for prop in ['g_vrh']:
        set_property_to_ids(non_ill_df, prop)
        res = main.main(args=args,torch_generator=1)
        resdict[k] = float(res)

        print(prop, resdict[k])

        st.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
        st.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
        st.move("test_results.csv", "./trained/" + prop + "_results.csv")
        
with open("train_outputs.json", "w") as f:
    json.dump(resdict, f)
print(time.time()-a)
