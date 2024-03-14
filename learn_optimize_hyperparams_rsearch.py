import pandas as pd
import json
import numpy as np
from subprocess import run
import shutil as st
import argparse
import main
import sys
import torch
import msgpack as mp

# df = pd.read_json("./data/root/data/mp.2018.6.1.json/mp.2018.6.1.json")
# df.info()
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
    "mp-ids-27430.csv": ["band_gap"],
    "mp-ids-46744.csv": ["energy_per_atom", "formation_energy_per_atom", "efermi"],
}

# print(args)
# set only the mp-ids that exist in the initial dataset

def load_properties_from_bin(file):
    with open(file, "rb") as f:
        properties = mp.unpackb(f.read())
        return properties
    return None

t2m_file = './data/root/data/t2m.bin'
t2m = load_properties_from_bin(t2m_file)

full_df = pd.read_csv("./scripts/non_ill_df.csv",index_col=0)
# print(full_df.head(10))

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

all_params = {
                # 'numprops':[2,3,10],
              'n_conv':list(range(1,6)),
              'atom_fea_len':[10,20,50,100,200],
              'h_fea_len':[10,20,50,100,200],
              'n_h':list(range(1,5)),
            #   '':list(range(1,6)),
            #   '':[np.exp(x) for x in [-6,-4,-2,0]],
            #   '':[np.exp(x) for x in [-8,-6,-4,-2]],
            #   '':[np.exp(x) for x in range(-8,-2)],
            #   '':[0,0.1,0.2]
            }
reslist = list()
maedict = dict()

current_property = 'band_gap'
# from sklearn.base import BaseEstimator
# class estimator(BaseEstimator):
#     def __init__(self, n_conv=0,atom_fea_len=0, h_fea_len=0,n_h=0):
#         self.n_conv =n_conv
#         self.atom_fea_len =atom_fea_len
#         self.h_fea_len =h_fea_len
#         self.n_h =n_h

#     def fit(self, X, y):
#         return self

#     def predict(self, X):
#         args.n_conv = self.n_conv
#         args.atom_fea_len = self.atom_fea_len
#         args.h_fea_len = self.h_fea_len
#         args.n_h = self.n_h
#         res = main.main(args=args,torch_generator=1)
#         st.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
#         st.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
#         st.move("test_results.csv", "./trained/" + prop + "_results.csv")
#         resdict[args] = float(res)
#         with open(f"opt_hyperparams_prop_{current_property}_log.json", "w") as f:
#             json.dump(resdict, f)
#         return float(res)

# def estimate(n_conv,atom_fea_len, h_fea_len,n_h):
#         args.n_conv =n_conv
#         args.atom_fea_len =atom_fea_len
#         args.h_fea_len =h_fea_len
#         args.n_h =n_h
#         res = main.main(args=args,torch_generator=1)
#         st.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
#         st.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
#         st.move("test_results.csv", "./trained/" + prop + "_results.csv")
#         resdict[args] = float(res)
#         with open(f"opt_hyperparams_prop_{current_property}_log.json", "w") as f:
#             json.dump(resdict, f)
#         return float(res)
import skopt
from skopt import space
# from sklearn.model_selection import RandomizedSearchCV
best_mae = dict()
# for csv in reference_csv.keys():
#     for prop in reference_csv[csv]:
prop = 'band_gap'
current_property = prop
# df = get_df_for_csv(csv)
df = get_df_for_csv('mp-ids-27430.csv')
print(df.describe())
set_property_to_ids(df, prop)
# random search 
# params = dict()
# for attempt in range(15):
#     for k,v in all_params.items():
#         params[k] = int(np.random.choice(v,1)[0])
#     print(params)
#     args.n_conv =params['n_conv']
#     args.atom_fea_len =params['atom_fea_len']
#     args.h_fea_len =params['h_fea_len']
#     args.n_h =params['n_h']
#     res = main.main(args=args,torch_generator=1)
#     st.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
#     st.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
#     st.move("test_results.csv", "./trained/" + prop + "_results.csv")
#     resdict[prop] = float(res), params
#     with open(f"opt_hyperparams_prop_{current_property}_log.json", "w") as f:
#         json.dump(resdict, f)
#     if prop not in best_mae:
#         best_mae[prop] = float(res),params
#     elif best_mae[prop][0]>float(res):
#         best_mae[prop]= float(res),params
SPACE = [
    space.Integer(1,6, name='n_conv', prior='uniform'),
    space.Integer(10, 50, name='atom_fea_len',prior='uniform'),
    space.Integer(10, 50, name='h_fea_len'),
    space.Integer(1, 6, name='n_h'),]
@skopt.utils.use_named_args(SPACE)
def objective(**params):
    args.n_conv =params['n_conv']
    args.atom_fea_len =params['atom_fea_len']
    args.h_fea_len =params['h_fea_len']
    args.n_h =params['n_h']
    res =  main.main(args= args,torch_generator=1)
    st.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
    st.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
    st.move("test_results.csv", "./trained/" + prop + "_results.csv")
    reslist.append((float(res), {
'n_conv':int(params['n_conv']),
'atom_fea_len':int(params['atom_fea_len']),
'h_fea_len':int(params['h_fea_len']),
'n_h':int(params['n_h']),}))
    with open(f"opt_hyperparams_prop_{current_property}_log.json", "w") as f:
        json.dump(reslist, f)
    return float(res)
from skopt import BayesSearchCV
results = skopt.forest_minimize(objective, SPACE, n_calls=100)

with open(f"opt_hyperparams_best_log.json", "w") as f:
    json.dump(best_mae, f)
import skopt.plots as plots
import matplotlib.pyplot as plt
plots.plot_convergence(results)
plt.savefig("fig.png")
