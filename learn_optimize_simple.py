import pandas as pd
import json
import numpy as np
from subprocess import run
import shutil as st
import argparse
import sys
import torch
import msgpack as mp
n_calls = 100

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
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
args = parser.parse_args()
args.data_options = ['./data/root/data/']
args.cuda = not args.disable_cuda and torch.cuda.is_available()
import main


# compress with zstd
import zstandard as zstd
cifs_file = './data/root/data/cifs.bin'
props_file = './data/root/data/props.bin'
cifs_compressed_file = './data/root/cifs.zstd'
props_compressed_file = './data/root/props.zstd'

def uncompress_all():
    with open(cifs_compressed_file, "rb") as f:
        res = zstd.decompress(f.read())
    with open(cifs_file, "wb") as f:
        f.write(res)
    with open(props_compressed_file, "rb") as f:
        res = zstd.decompress(f.read())
    with open(props_file, "wb") as f:
        f.write(res)

uncompress_all()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


fields = [
    "energy_per_atom",
    "formation_energy_per_atom",
    "band_gap",
    "efermi",
    # "k_voigt",
    # "k_reuss",
    # "k_vrh",
    # "g_voigt",
    # "g_reuss",
    # "g_vrh",
    "homogeneous_poisson",
]

reference_csv = {
    "mp-ids-3402.csv": [
        # "k_voigt",
        # "k_reuss",
        # "k_vrh",
        # "g_voigt",
        # "g_reuss",
        # "g_vrh",
        "homogeneous_poisson",
    ],
    "mp-ids-46744.csv": ["energy_per_atom", "formation_energy_per_atom", "efermi"],
    "mp-ids-27430.csv": ["band_gap"],
}

def load_properties_from_bin(file):
    with open(file, "rb") as f:
        properties = mp.unpackb(f.read())
        return properties
    return None

t2m_file = './data/root/data/t2m.bin'
t2m = load_properties_from_bin(t2m_file)

full_df = pd.read_csv("./scripts/non_ill_df.csv",index_col=0)

full_df = pd.DataFrame(load_properties_from_bin('./data/root/data/props.bin')).transpose()
# print(full_df.head(10))

for moduli in ["k_voigt", "k_reuss", "k_vrh", "g_voigt", "g_reuss", "g_vrh"]:
    full_df[moduli] = np.log(full_df[moduli])
# print(full_df.describe())
def get_df_for_csv(csv:str):
    global t2m, full_df

    ids = pd.read_csv("./data/material-data/" + csv)
    ids = [list(ids)[0]]+list(ids.iloc[:, 0])
    df = pd.DataFrame()
    for t in ids:
        m = t2m[t]
        df.loc
        df=pd.concat([df,full_df.loc[[m]]])
    return df

def set_property_to_ids(df: pd.DataFrame, prop: str):
    part_df = df[prop].dropna()
    part_df = part_df.groupby(part_df.index).first()
    part_df.to_csv("./data/root/data/id_prop.csv", index=True, header=False)


reslist = list()

import skopt
from skopt import space
best_mae = dict()
prop = 'band_gap'
current_property = prop
df = get_df_for_csv('mp-ids-27430.csv')
print(df.describe())
set_property_to_ids(df, prop)
SPACE = [
    space.Integer(1,6, name='n_conv', prior='uniform'),
    space.Integer(10, 200, name='atom_fea_len',prior='uniform'),
    space.Integer(10, 200, name='h_fea_len'),
    space.Integer(1, 6, name='n_h'),]
@skopt.utils.use_named_args(SPACE)
def objective(**params):
    tmp = {
'n_conv':int(params['n_conv']),
'atom_fea_len':int(params['atom_fea_len']),
'h_fea_len':int(params['h_fea_len']),
'n_h':int(params['n_h']),}
    reslist.append((100,tmp))
    with open(f"optHP_{current_property}_log.json", "w") as f:
        json.dump(reslist, f)
    
    args.n_conv =params['n_conv']
    args.atom_fea_len =params['atom_fea_len']
    args.h_fea_len =params['h_fea_len']
    args.n_h =params['n_h']
    oom = False
    try:
        res =  main.main(args= args,torch_generator=1)
        st.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
        st.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
        st.move("test_results.csv", "./trained/" + prop + "_results.csv")
    except RuntimeError:
        oom = True
        
    if oom:
        with torch.no_grad():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        args.disable_cuda =True
        args.cuda = not args.disable_cuda and torch.cuda.is_available()
        # res =  main.main(args= args,torch_generator=1)
        res = 4.2
        # <- ~ big value
        args.disable_cuda =False
        args.cuda = not args.disable_cuda and torch.cuda.is_available()

    res=float(res)
    with torch.no_grad():
        torch.cuda.empty_cache()
    reslist[-1] = ((res,tmp))
    with open(f"optHP_{current_property}_log.json", "w") as f:
        json.dump(reslist, f)
    return res

for k,v in reference_csv.items():
    for prop in v:
        reslist = list()
        # best_mae = dict()
        current_property = prop
        df = get_df_for_csv(k)
        print(df.describe())
        set_property_to_ids(df, prop)

        results = skopt.gp_minimize(objective, SPACE, n_calls=n_calls, n_initial_points=1,x0=[[i.high for i in SPACE]])
        # with open(f"optHP_bestmae_{current_property}_log.json", "w") as f:
            # json.dump(best_mae, f)
        import skopt.plots as plots
        import matplotlib.pyplot as plt
        _ = plots.plot_objective(results,  sample_source='result', n_points=20)
        plt.savefig(f"objective_{current_property}_.png")