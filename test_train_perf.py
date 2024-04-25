import os
from pathlib import Path
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester
import pandas as pd
import numpy as np
from matplotlib_venn import venn2

# dataset saving paths
basedir = "./data/root/data/"
t2m_file = "./data/root/data/t2m.json"

def load_from_json(file):
    with open(file, "rb") as f:
        import orjson

        return orjson.loads(f.read())
    
 # some thermodynamic properties of Materials Project significantly changed recently due to migration from GGA/GGA+U to r2SCAN
dataset_original_before_r2scan_path = basedir + "dataset_original_before_r2scan/"
dataset_full_before_r2scan_path = (
    basedir + "dataset_full_before_r2scan/"
)  # does not contain shear/bulk modulus

dataset_original_path = basedir + 'dataset_original_2024-04-18/'
dataset_full_path = basedir + 'dataset_full_2024-04-18/'
dataset_delta_path = basedir + 'dataset_delta/'

from typing import Callable

# optional connect to clearML
# %env CLEARML_WEB_HOST="https://app.clear.ml"
# %env CLEARML_API_HOST="https://api.clear.ml"
# %env CLEARML_FILES_HOST="https://files.clear.ml"
# %env CLEARML_API_ACCESS_KEY="..."
# %env CLEARML_API_SECRET_KEY="..."
# import clearml
# clearml.browser_login()
is_clearml = True

# train configuration
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
    # "mp-ids-27430.csv": ["band_gap"],
    # "mp-ids-46744.csv": ["energy_per_atom", "formation_energy_per_atom", "efermi"],
}

fields = [
    # "energy_per_atom",
    # "formation_energy_per_atom",
    # "band_gap",
    # "efermi",
    # "k_voigt",
    # "k_reuss",
    # "k_vrh",
    # "g_voigt",
    # "g_reuss",
    # "g_vrh",
    "homogeneous_poisson",
]

dataset_vs_train_fields = [
    (dataset_original_path, reference_csv),
    (dataset_original_path, fields),
    (dataset_full_path, fields),
    (dataset_delta_path, fields),
    (dataset_original_before_r2scan_path, fields),
    (dataset_full_before_r2scan_path, fields),
]

import pandas as pd

def set_property_to_ids(df: pd.DataFrame, property: str, csv :str = "./data/root/data/id_prop.csv"):
    df[property].dropna().to_csv(
        csv, index=True, header=False
    )

t2m = load_from_json(t2m_file)

def get_df_for_csv(csv: str, full_df:pd.DataFrame):
    global t2m

    ids = pd.read_csv("./data/material-data/" + csv)
    ids = [list(ids)[0]] + list(ids.iloc[:, 0])
    df = pd.DataFrame()
    idx = full_df.index
    ms = [t2m[t] for t in ids if t2m[t] in idx]
    df = full_df.loc[ms]
    return df

# train selected properties on all datasets with default hyperparameters
def train_default():
    """train with default hyperparameters"""

    import main

    mae = main.main(free_cache=True)
    # del main
    return mae


def clearml_train_logger(ds_path: str, train_fn: Callable[[str], float]):
    ds_name = Path(ds_path).name
    if not is_clearml:
        mae = train_fn()
        print(ds_name, mae)
    else:
        # prepare task
        from clearml import Task
        Task.set_offline(True)

        task: Task = Task.init(
            project_name="rcgcnn",
            task_name="train " + ds_path,
        )
        # Setting the custom parameters
        params = {
            "ds_path": ds_path,
            "ds_name": ds_name,
        }
        task.connect(params)
        # from torch.utils.tensorboard import SummaryWriter

        # writer = SummaryWriter()
        mae = train_fn()
        # finish task
        # save model file
        task.upload_artifact(name='model_best.pth.tar', artifact_object='model_best.pth.tar')

        # parse console output into clearml scalars
        console_output = task.get_reported_console_output()

        for line in console_output:
            print(line)
        # writer.add_scalar('Training loss', loss.item(), epoch * len(train_loader) + i)
        result_info = {
            "result_mae": mae,
        }
        task.connect(result_info)

        task.close()


def train_all_datasets(
    dataset_vs_train_fields: list[tuple[str, list | dict]],
    clearml_train_logger: Callable[[str, Callable[[str], float]], float],
    train_fn: Callable[[], float],
):
    for ds_path, ds_props in dataset_vs_train_fields:
        full_df = pd.DataFrame(load_from_json(ds_path + "props.json")).transpose()
        print(full_df.describe())
        # clear sys argv for argparse
        import sys
        sys.argv = [ds_path,ds_path]
        del sys

        import shutil as sh
        sh.copy(basedir+'atom_init.json',ds_path)

        # choose ds prop format:
        if type(ds_props) is dict:
            for csv in ds_props.keys():
                for prop in ds_props[csv]:
                    # set dataset - property pair
                    df = get_df_for_csv(csv, full_df)
                    set_property_to_ids(df, prop, ds_path + "id_prop.csv")
                    # remove old model files
                    import shutil as sh
                    try:
                        sh.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
                        sh.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
                        sh.move("test_results.csv", "./trained/" + prop + "_results.csv")
                    except:
                        pass
                    clearml_train_logger(ds_path, train_fn)
                    # print('no', prop,'in,ds_path)
        elif type(ds_props) is dict:
            for prop in ds_props:
                # set dataset - property pair
                set_property_to_ids(full_df, prop, ds_path + "id_prop.csv")
                # remove old model files
                import shutil as sh
                try:
                    sh.move("checkpoint.pth.tar", "./trained/" + prop + "_check.pth.tar")
                    sh.move("model_best.pth.tar", "./trained/" + prop + "_best.pth.tar")
                    sh.move("test_results.csv", "./trained/" + prop + "_results.csv")
                except:
                    pass
                clearml_train_logger(ds_path, train_fn)

train_all_datasets(dataset_vs_train_fields,clearml_train_logger, train_default)