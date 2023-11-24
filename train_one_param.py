import os
from pathlib import Path
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester
import pandas as pd
import json
import ormsgpack as mp
import numpy as np
from matplotlib_venn import venn2
api_key=os.environ.get('MP_API_KEY')
ids = pd.read_csv("./data/material-data/mp-ids-46744.csv")
ids = [list(ids)[0]]+list(ids.iloc[:, 0])
print(len(ids))


# print data statistics for first dataset
basedir = './data/root/data/'

# propfiles = os.listdir(basedir)
# def load_properties_from_json(file):
#     filename = os.path.basename(file)
#     with open(basedir + file, "rb") as f:
#         properties = json.load(f)
#         return (filename, properties)
#     return None


def load_properties_from_bin(file):
    with open(file, "rb") as f:
        properties = mp.unpackb(f.read())
        return properties
    return None

# propdata = dict()
# for file in propfiles:
#     name,prop=load_properties_from_json(file)
#     for key in prop.keys():
#         propdata.setdefault(key,[])
#         propdata[key].append(prop[key])

full_df = pd.DataFrame(load_properties_from_bin(basedir + "props.bin")).transpose()
import numpy as np

for moduli in ["k_voigt", "k_reuss", "k_vrh", "g_voigt", "g_reuss", "g_vrh"]:
    full_df[moduli] = np.log(full_df[moduli])
full_df.describe()
# f.to_csv('./test_missing.csv')

import numpy as np
from subprocess import run
import shutil as st

resdict = dict()
maedict = dict()
reference_csv = {
    "mp-ids-3402.csv": [
        "k_voigt",
        "k_reuss",
        "k_vrh",
        "g_voigt",
        "g_reuss",
        "g_vrh",
        "homogeneous_poisson",
    ],
    "mp-ids-27430.csv": ["band_gap"],
    "mp-ids-46744.csv": ["energy_per_atom", "formation_energy_per_atom", "efermi"],
}


def set_property_to_ids(df: pd.DataFrame, property: str):
    # nids = df[property].isna()
    df[property].dropna().to_csv("./data/root/data/id_prop.csv", index=True, header=False)

def prepare_csv(csv: str, df: pd.DataFrame, prop: str):
    # st.move("./data/material-data/" + csv, "./data/root/data/id_prop.csv")
    set_property_to_ids(df, prop)

t2m_file = './data/root/data/t2m.bin'
t2m = load_properties_from_bin(t2m_file)

def get_df_for_csv(csv:str):
    global t2m, full_df
    ids = pd.read_csv("./data/material-data/" + csv)
    ids = [list(ids)[0]]+list(ids.iloc[:, 0])
    df = pd.DataFrame()
    for t in ids:
        m = t2m[t]
        df=pd.concat([df,full_df.loc[[m]]])
    return df

csv = 'mp-ids-3402.csv'
df = get_df_for_csv(csv)
prop = 'k_vrh'
prepare_csv(csv, df, prop)
import main
for k in range(2):
    mae = main.main(data_options='./data/root/data/')
    # result = run(['ls'], capture_output=True, text=True)
    # Write the contents of the stdout (standard output) to the file.
    resdict[(prop,k)]=mae
    print(resdict)
    st.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
    st.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
    st.move("test_results.csv", "./trained/" + prop + "_results.csv")
    
    with open("train_outputs.json", "w") as f:
        json.dump(resdict, f)
    # pd.DataFrame(maedict, index=['MAE']).transpose().to_csv('train_maes.csv')

for k in range(2):
    mae = main.main(data_options='./data/root/data/',torch_generator=1)
    # result = run(['ls'], capture_output=True, text=True)
    # Write the contents of the stdout (standard output) to the file.
    resdict[(prop,k)]=mae
    print(resdict)
    st.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
    st.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
    st.move("test_results.csv", "./trained/" + prop + "_results.csv")
    
    with open("train_outputs.json", "w") as f:
        json.dump(resdict, f)
    # pd.DataFrame(maedict, index=['MAE']).transpose().to_csv('train_maes.csv')