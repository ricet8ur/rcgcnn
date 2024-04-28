from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)

    # replace Random Sampler
    if 'torch_generator' in kwargs:
        print('torch_gen')
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # g = torch.Generator()
        # g=g.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        g=torch.manual_seed(42)
        # torch.cuda.manual_seed_all(42)
    else:
        g=None
    train_sampler = SubsetRandomSampler(indices[:train_size],g)
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size],g)
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:],g)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory,generator=g)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory,generator=g)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory,generator=g)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids

import numba as nb
@nb.njit
def _expand(self_filter:np.ndarray, self_var:float, distances:np.ndarray):
    return np.exp(-(distances[..., np.newaxis] - self_filter)**2 / self_var**2)

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        # slower
        # return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                    #   self.var**2)
        # faster:
        return _expand(self.filter,self.var,distances)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


# inner pymatgen core functions reimplementation
from pymatgen.core.sites import PeriodicSite, Site
from collections.abc import Iterable, Iterator, Sequence
from pymatgen.core.structure import PeriodicNeighbor

def get_all_neighbors(
    self:Structure,
    r: float,
    include_index: bool = False,
    include_image: bool = False,
    sites: Sequence[PeriodicSite] | None = None,
    numerical_tol: float = 1e-8,
) -> list[list[PeriodicNeighbor]]:
    import collections
    from pymatgen.core.sites import PeriodicSite, Site
    from collections.abc import Iterable, Iterator, Sequence
    from pymatgen.core.structure import PeriodicNeighbor
    """Get neighbors for each atom in the unit cell, out to a distance r
    Returns a list of list of neighbors for each site in structure.
    Use this method if you are planning on looping over all sites in the
    crystal. If you only want neighbors for a particular site, use the
    method get_neighbors as it may not have to build such a large supercell
    However if you are looping over all sites in the crystal, this method
    is more efficient since it only performs one pass over a large enough
    supercell to contain all possible atoms out to a distance r.
    The return type is a [(site, dist) ...] since most of the time,
    subsequent processing requires the distance.
    A note about periodic images: Before computing the neighbors, this
    operation translates all atoms to within the unit cell (having
    fractional coordinates within [0,1)). This means that the "image" of a
    site does not correspond to how much it has been translates from its
    current position, but which image of the unit cell it resides.
    Args:
        r (float): Radius of sphere.
        include_index (bool): Deprecated. Now, the non-supercell site index
            is always included in the returned data.
        include_image (bool): Deprecated. Now the supercell image
            is always included in the returned data.
        sites (list of Sites or None): sites for getting all neighbors,
            default is None, which means neighbors will be obtained for all
            sites. This is useful in the situation where you are interested
            only in one subspecies type, and makes it a lot faster.
        numerical_tol (float): This is a numerical tolerance for distances.
            Sites which are < numerical_tol are determined to be coincident
            with the site. Sites which are r + numerical_tol away is deemed
            to be within r from the site. The default of 1e-8 should be
            ok in most instances.
    Returns:
        [[pymatgen.core.structure.PeriodicNeighbor], ..]
    """
    if sites is None:
        sites = self.sites
    center_indices, points_indices, images, distances = self.get_neighbor_list(
        r=r, sites=sites, numerical_tol=numerical_tol
    )
    if len(points_indices) < 1:
        return [[]] * len(sites)
    f_coords = self.frac_coords[points_indices] + images
    neighbor_dict: dict[int, list] = collections.defaultdict(list)
    lattice = self.lattice
    atol = Site.position_atol
    all_sites = self.sites
    for cindex, pindex, image, f_coord, d in zip(center_indices, points_indices, images, f_coords, distances):
        psite = all_sites[pindex]
        csite = sites[cindex]
        if (
            d > numerical_tol
            or
            # This simply compares the psite and csite. The reason why manual comparison is done is
            # for speed. This does not check the lattice since they are always equal. Also, the or construct
            # returns True immediately once one of the conditions are satisfied.
            psite.species != csite.species
            or (not np.allclose(psite.coords, csite.coords, atol=atol))
            or (psite.properties != csite.properties)
        ):
            neighbor_dict[cindex].append(
                {1:d,2:pindex}
                # PeriodicNeighbor(
                #     species=psite.species,
                #     coords=f_coord,
                #     lattice=lattice,
                #     properties=psite.properties,
                #     nn_distance=d,
                #     index=pindex,
                #     image=tuple(image),
                #     label=psite.label,
                # )
            )
    # neighbors: list[list[PeriodicNeighbor]] = []
    neighbors: list[list[dict]] = []
    for i in range(len(sites)):
        neighbors.append(neighbor_dict[i])
    return neighbors 

import multiprocessing
class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, max_cache_size=80000):

        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        # create new bijective map between mp-id and index
        # based on sorted indexes of mp-ids for the given main() call
        # print('p',self.id_prop_data)
        self.idx2mid = dict()
        self.mid2target = dict() # aka new self.id_prop_data with mp_id index instead of random index
        mp_ids_list = []
        for idx in range(len(self.id_prop_data)):
            cif_id, target = self.id_prop_data[idx]
            mid = int(cif_id[3:])
            mp_ids_list.append(mid)
            self.mid2target[mid] = (cif_id,target)

        self.idx_sequence = []
        for idx, mid in enumerate(sorted(mp_ids_list)):
            self.idx2mid[idx]=mid

        self.idx_sequence=[idx for idx in range(len(self.id_prop_data))]
        # instead of self.id_prop_data operate on suffled idx_sequence=[idx1,idx2,idx3]
        random.seed(random_seed)
        random.shuffle(self.idx_sequence)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        # caching
        self.manager = multiprocessing.Manager()
        self.max_cache_size=max_cache_size
        self.rlock = self.manager.RLock()
        # if self.shared_dict is None:
        #     print('error: shared_dict is None')
        import orjson as json
        with open(os.path.join(self.root_dir,'cifs.json'),'rb') as f:
            # load all cifs from 'cifs.json', which contains cif file for each possible mp-id
            self.cifs = self.manager.dict([(k,v) for k,v in json.loads(f.read()).items()])
        self.shared_dict = self.manager.dict()
        # self.shared_cache = manager.dict()
        # simple cache that support DataLoader with multiple workers
        # + using fix from https://discuss.pytorch.org/t/pytorch-cannot-allocate-memory/134754/19 
        # and https://latentwalk.io/2023/08/19/torch-shmem/

    def __len__(self):
        return len(self.idx_sequence)

    # @functools.lru_cache(maxsize=100000)  # Cache loaded structures - does not use multiprocessing 
    # - unable to use multiple workers:
    # https://stackoverflow.com/questions/43495986/combining-functools-lru-cache-with-multiprocessing-pool

    # @my_lru_cache(maxsize=50000)
    def __getitem__(self, idx):
        # if len(self.cifs) == self.__getitem__.cache_info()[2]:
            # able to cache all features, so can free cifs
            # del self.cifs
            # pass
        mid = self.idx2mid[self.idx_sequence[idx]]
        to_calculate=True
        with self.rlock:
            if mid in self.shared_dict:
                to_calculate = False
        
        if to_calculate: 
        # if len(self.shared_dict) > len(self.cifs):
            #     # pop random feature from cache to keep its size constant
            #     amount_to_pop = 10
            #     for random_idx in np.random.choice(list(self.cifs.keys()), amount_to_pop):
            #         if random_idx in self.shared_dict:
            #             self.shared_dict.pop(random_idx)

            cif_id, target = self.mid2target[mid]
            crystal = Structure.from_dict(self.cifs[cif_id])
            atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                                  for i in range(len(crystal))])
            atom_fea = torch.Tensor(atom_fea)
            all_nbrs = get_all_neighbors(crystal, self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    warnings.warn('{} not find enough neighbors to build graph. '
                                  'If it happens frequently, consider increase '
                                  'radius.'.format(cif_id))
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                       [0] * (self.max_num_nbr - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                   [self.radius + 1.] * (self.max_num_nbr -
                                                         len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2],
                                                nbr[:self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1],
                                            nbr[:self.max_num_nbr])))
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)
            atom_fea = torch.Tensor(atom_fea)
            nbr_fea = torch.Tensor(nbr_fea)
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
            target = torch.Tensor([float(target)])
            data = ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
            with self.rlock:
                self.shared_dict[mid] = data
        return self.shared_dict[mid]