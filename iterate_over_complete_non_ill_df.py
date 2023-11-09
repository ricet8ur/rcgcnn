import pandas as pd
import json
import numpy as np
from subprocess import run
import shutil as st

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
print(non_ill_df.head())
resdict = dict()
maedict = dict()
for prop in fields:
    set_property_to_ids(non_ill_df, prop)
    res = run(
        "conda run -n cgcnn2 python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 ./data/root/data/",
        capture_output=True,
        text=True,
        shell=True,
    )
    resdict[prop] = res.stdout
    maedict[prop] = resdict[prop].split("** MAE ")[-1].replace('\n','')
    
    print(prop, maedict[prop])

    st.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
    st.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
    st.move("test_results.csv", "./trained/" + prop + "_results.csv")
        
with open("train_outputs.json", "w") as f:
    json.dump(resdict, f)
pd.DataFrame(maedict, index=['MAE']).to_csv('train_maes.csv')