import os
import random

import torch

from modules.bio.util import *


class ProteinMutations(torch.utils.data.IterableDataset):

    def __init__(self, df, atoms_mapper=None, pdb_root='pdb', max_atoms=3500, pe_features=16):
        super(ProteinMutations, self).__init__()

        if atoms_mapper is None:
            atoms_mapper = {'ND2': 0, 'CZ3': 1, 'SD': 2, 'NH1': 3, 'SG': 4, 'CG1': 5, 'OG1': 6, 'CB': 7, 'CH2': 8,
                            'CE': 9, 'OE1': 10, 'CZ': 11, 'OE2': 12, 'CZ2': 13, 'C': 14, 'OH': 15, 'CG': 16,
                            'CD1': 17, 'CA': 18, 'OD2': 19, 'NZ': 20, 'CE1': 21, 'CE3': 22, 'NH2': 23, 'NE': 24,
                            'ND1': 25, 'CE2': 26, 'OD1': 27, 'CD': 28, 'NE1': 29, 'OG': 30, 'CG2': 31, 'NE2': 32,
                            'O': 33, 'N': 34, 'CD2': 35}
        self.df = df
        self.pdb_root = pdb_root
        self.max_atoms = max_atoms
        self.pe_features = pe_features

        if len(atoms_mapper) == 0:
            paths = list(set(df['pdb_path'].unique().tolist() + df['pdb_path'].unique().tolist()))
            unique_atoms = list(set([atom for path in paths for atom in
                                     PandasPdb().read_pdb(f'{pdb_root}/{path}').df['ATOM'][
                                         'atom_name'].unique().tolist()]))
            atoms_mapper = {name: idx for name, idx in zip(unique_atoms, range(len(unique_atoms)))}
        self.atoms_mapper = atoms_mapper

    def __iter__(self):
        return self

    def __next__(self):
        idx = random.randint(0, len(self.df) - 1)
        row = self.df.iloc[idx]
        wt_points, wt_features, wt_atom_ids, wt_mask = get_pdb_features(
            os.path.join(self.pdb_root, row['pdb_path']), self.atoms_mapper, self.max_atoms, self.pe_features)
        mut_points, mut_features, mut_atom_ids, mut_mask = get_pdb_features(
            os.path.join(self.pdb_root, row['mut_path']), self.atoms_mapper, self.max_atoms, self.pe_features)
        return {
            'wt_points': wt_points,
            'wt_features': wt_features,
            'wt_atom_ids': wt_atom_ids,
            'wt_mask': wt_mask,
            'mut_points': mut_points,
            'mut_features': mut_features,
            'mut_atom_ids': mut_atom_ids,
            'mut_mask': mut_mask,
            'dT': row['dT'].astype(np.float32),
            'pH': row['pH'].astype(np.float32)
        }
