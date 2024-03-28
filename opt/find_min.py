import json
import os
import pathlib
files = os.listdir('./')
for f in files:
    p = pathlib.Path(f)
    if p.suffix == '.json':
        with open('./'+f) as s:
            d = json.loads(s.read())
            m = min([l[0] for l in d])
            print(p.name, d[[l[0] for l in d].index(m)])

# optHP_band_gap_log.json 0.46777060627937317
# optHP_efermi_log.json 0.4299956262111664
# optHP_energy_per_atom_log.json 0.11319795995950699
# optHP_formation_energy_per_atom_log.json 0.06707489490509033
# optHP_g_reuss_log.json 0.2643553912639618
# optHP_g_voigt_log.json 0.17695805430412292
# optHP_g_vrh_log.json 0.20536403357982635
# optHP_homogeneous_poisson_log.json 0.028519265353679657
# optHP_k_reuss_log.json 0.1848583221435547
# optHP_k_voigt_log.json 0.12643878161907196
# optHP_k_vrh_log.json 0.14375068247318268


# band_gap_log [0.46777060627937317, {'n_conv': 6, 'atom_fea_len': 139, 'h_fea_len': 5, 'n_h': 2}]
# efermi_log [0.4299956262111664, {'n_conv': 6, 'atom_fea_len': 146, 'h_fea_len': 151, 'n_h': 2}]
# energy_per_atom_log [0.11319795995950699, {'n_conv': 6, 'atom_fea_len': 200, 'h_fea_len': 101, 'n_h': 3}]
# formation_energy_per_atom_log [0.06707489490509033, {'n_conv': 6, 'atom_fea_len': 158, 'h_fea_len': 200, 'n_h': 3}]
# g_reuss_log [0.2643553912639618, {'n_conv': 4, 'atom_fea_len': 38, 'h_fea_len': 94, 'n_h': 2}]
# g_voigt_log [0.17695805430412292, {'n_conv': 6, 'atom_fea_len': 113, 'h_fea_len': 31, 'n_h': 3}]
# g_vrh_log [0.20536403357982635, {'n_conv': 2, 'atom_fea_len': 157, 'h_fea_len': 18, 'n_h': 3}]
# homogeneous_poisson_log [0.028519265353679657, {'n_conv': 2, 'atom_fea_len': 200, 'h_fea_len': 200, 'n_h': 3}]
# k_reuss_log [0.1848583221435547, {'n_conv': 5, 'atom_fea_len': 57, 'h_fea_len': 115, 'n_h': 4}]
# k_voigt_log [0.12643878161907196, {'n_conv': 4, 'atom_fea_len': 200, 'h_fea_len': 200, 'n_h': 3}]
# k_vrh_log [0.14375068247318268, {'n_conv': 4, 'atom_fea_len': 141, 'h_fea_len': 90, 'n_h': 4}]


# band_gap {'n_conv': 6, 'atom_fea_len': 139, 'h_fea_len': 5, 'n_h': 2}
# efermi {'n_conv': 6, 'atom_fea_len': 146, 'h_fea_len': 151, 'n_h': 2}
# energy_per_atom {'n_conv': 6, 'atom_fea_len': 200, 'h_fea_len': 101, 'n_h': 3}
# formation_energy_per_atom {'n_conv': 6, 'atom_fea_len': 158, 'h_fea_len': 200, 'n_h': 3}
# g_reuss {'n_conv': 4, 'atom_fea_len': 38, 'h_fea_len': 94, 'n_h': 2}
# g_voigt {'n_conv': 6, 'atom_fea_len': 113, 'h_fea_len': 31, 'n_h': 3}
# g_vrh {'n_conv': 2, 'atom_fea_len': 157, 'h_fea_len': 18, 'n_h': 3}
# homogeneous_poisson {'n_conv': 2, 'atom_fea_len': 200, 'h_fea_len': 200, 'n_h': 3}
# k_reuss {'n_conv': 5, 'atom_fea_len': 57, 'h_fea_len': 115, 'n_h': 4}
# k_voigt {'n_conv': 4, 'atom_fea_len': 200, 'h_fea_len': 200, 'n_h': 3}
# k_vrh {'n_conv': 4, 'atom_fea_len': 141, 'h_fea_len': 90, 'n_h': 4}