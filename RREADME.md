# Reproduction of Crystal Graph Convolutional Neural Networks
Create conda env for linux machine:

```bash
conda create -n cgcnn python=3.11
conda activate cgcnn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ipykernel pymatgen==2023.10.4 mp-api scikit-learn ormsgpack zstandard pyflame matplotlib-venn
```

Pymatgen newer version prints too many warnings during cif read. + For some reason prints len of cif structures for every structure...
Ormsgpack is needed to replace ~40k .cif files with one .bin