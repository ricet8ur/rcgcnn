# compace materials projects df from 2018 and now
import msgpack as mp
import pandas as pd
import numpy as np
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

# load new_df
basedir = '../data/root/data/'
def load_properties_from_bin(file):
    with open(file, "rb") as f:
        properties = mp.unpackb(f.read())
        return properties
    return None

full_df = pd.DataFrame(load_properties_from_bin(basedir + "props.bin")).transpose()

for moduli in ["k_voigt", "k_reuss", "k_vrh", "g_voigt", "g_reuss", "g_vrh"]:
    full_df[moduli] = np.log(full_df[moduli])
print(full_df.describe())

# load old df
mp2018path = '../data/root/data/mp.2018.6.1.json/mp.2018.6.1.json'

old_df = pd.read_json(mp2018path)
print(old_df.describe())

# load task_id to mp_id mapping 
t2m_file = '../data/root/data/t2m.bin'
t2m = load_properties_from_bin(t2m_file)

# just reference code from 'learn.ipynb'
# def get_df_for_csv(csv:str):
#     global t2m, full_df
#     ids = pd.read_csv("./data/material-data/" + csv)
#     ids = [list(ids)[0]]+list(ids.iloc[:, 0])
#     df = pd.DataFrame()
#     for t in ids:
#         m = t2m[t]
#         df=pd.concat([df,full_df.loc[[m]]])
#     return df

# translate task_ids in old dataset to mp_ids via t2m map
old_df=old_df.set_index('material_id')
# ids = list(old_df.index)
# print(ids)
# df = pd.DataFrame()
# for t in ids:
#     if t in t2m:
#         m = t2m[t]
#         old_df.rename(index={t:m})
#         # print(t in old_df.index)
#         # part = old_df.loc[t,:]
#         # print(part)
#         # part.rename(index={t:m})
#         # print(part)
#         # df=pd.concat([df,part])
#         # break
df = old_df.rename(index=lambda t:t2m[t] if t in t2m else t)
print(df.describe())
# compare indexes via venn diagrams
    
# my venn diagram for two sets
def venn(**kvarg):
    subsets = []
    set_labels = []
    for k,v in kvarg.items():
        subsets.append(set(v))
        set_labels.append(k)
    return venn2(subsets=subsets,set_labels=set_labels)

ids_new = list(full_df.index)
ids_old = list(df.index)
# print(ids_new[:5])
# print(ids_old[:5])
venn(new=ids_new,old=ids_old)
plt.savefig('venn new vs 2018.png')

# compare properties
prop_new = 'k_voigt'
prop_old = 'K'

prop_slice_new = full_df[[prop_new]]
prop_slice_old = df[[prop_old]]

prop_slice_new.describe()
prop_slice_old.describe()
ids_new = list(prop_slice_new.index)
ids_old = list(prop_slice_old.index)
# filtering by same indexes:
same_ids = set(ids_new).intersection(ids_old)

new_filtered = prop_slice_new[prop_slice_new.index.isin(same_ids)]
old_filtered = prop_slice_old[prop_slice_old.index.isin(same_ids)]

# plot
same_ids = list(same_ids)
dict_new = new_filtered.to_dict()[prop_new]
dict_old = old_filtered.to_dict()[prop_old]
list_new = [dict_new[idx] for idx in same_ids if dict_new[idx] is not None and not np.isnan(dict_new[idx]) and (dict_old[idx] is not None and not np.isnan(dict_old[idx]))]    
list_old = [dict_old[idx] for idx in same_ids if dict_old[idx] is not None and not np.isnan(dict_old[idx]) and (dict_new[idx] is not None and not np.isnan(dict_new[idx]))]    
# print(list_new[:10])
proptype = 'moduli'
if proptype=='moduli':
    list_old=[np.log(x) for x in list_old]
# print(list_new)
# print(list_old)
same_list = [list_new[idx] for idx in range(len(list_new)) if list_new[idx]==list_old[idx]]
different_list = [list_new[idx] for idx in range(len(list_new)) if list_new[idx]!=list_old[idx]]
# print(len(different_list),
plt.close()
plt.plot(list_new[::100],label=f'new (differences: {len(different_list)})')
plt.plot(list_old[::100],label=f'old (similarities: {len(same_list)})')
plt.legend()
plt.savefig(f'Prop: {prop_new} comparison 2018-now.png')