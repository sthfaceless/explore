import multiprocessing

from tqdm import tqdm

from modules.bio.util import *


class ProteinMutations(torch.utils.data.IterableDataset):

    def __init__(self, df, true_indexes=None, atoms_mapper=None, acid_mapper=None, pdb_root='pdb', max_atoms=3500,
                 seq_len=400, pe_features=16, precompute=None, precomputed=None):
        super(ProteinMutations, self).__init__()

        if atoms_mapper is None:
            atoms_mapper = {'ND2': 0, 'CZ3': 1, 'SD': 2, 'NH1': 3, 'SG': 4, 'CG1': 5, 'OG1': 6, 'CB': 7, 'CH2': 8,
                            'CE': 9, 'OE1': 10, 'CZ': 11, 'OE2': 12, 'CZ2': 13, 'C': 14, 'OH': 15, 'CG': 16,
                            'CD1': 17, 'CA': 18, 'OD2': 19, 'NZ': 20, 'CE1': 21, 'CE3': 22, 'NH2': 23, 'NE': 24,
                            'ND1': 25, 'CE2': 26, 'OD1': 27, 'CD': 28, 'NE1': 29, 'OG': 30, 'CG2': 31, 'NE2': 32,
                            'O': 33, 'N': 34, 'CD2': 35}
        if len(atoms_mapper) == 0:
            paths = list(set(df['pdb_path'].unique().tolist() + df['pdb_path'].unique().tolist()))
            unique_atoms = list(set([atom for path in paths for atom in
                                     PandasPdb().read_pdb(f'{pdb_root}/{path}').df['ATOM'][
                                         'atom_name'].unique().tolist()]))
            atoms_mapper = {name: idx for name, idx in zip(unique_atoms, range(len(unique_atoms)))}
        self.atoms_mapper = atoms_mapper

        if acid_mapper is None:
            self.acid_mapper = {'T': 0, 'P': 1, 'K': 2, 'Q': 3, 'E': 4, 'W': 5, 'C': 6, 'M': 7, 'Y': 8, 'D': 9, 'L': 10,
                                'V': 11, 'F': 12, 'A': 13, 'G': 14, 'R': 15, 'N': 16, 'S': 17, 'H': 18, 'I': 19}
        self.df = df
        self.true_indexes = true_indexes
        self.pdb_root = pdb_root
        self.max_atoms = max_atoms
        self.seq_len = seq_len
        self.pe_features = pe_features

        self.precompute = precompute
        if precompute is not None:
            os.makedirs(precompute, exist_ok=True)
            with multiprocessing.Pool() as pool:
                pool.map(self.save_features, tqdm(range(len(self.df))))

        self.precomputed = precomputed

    def calc_features(self, idx):
        row = self.df.iloc[idx]
        wt_points, wt_features, wt_atom_ids, wt_mask, wt_alpha_points, wt_alpha_mask = get_pdb_features(
            os.path.join(self.pdb_root, row['pdb_path']), self.atoms_mapper, self.max_atoms, self.seq_len,
            self.pe_features)
        mut_points, mut_features, mut_atom_ids, mut_mask, mut_alpha_points, mut_alpha_mask = get_pdb_features(
            os.path.join(self.pdb_root, row['mut_path']), self.atoms_mapper, self.max_atoms, self.seq_len,
            self.pe_features)
        # rotate mutation protein
        rotation, transform = get_protein_transform(wt_alpha_points[wt_alpha_mask], mut_alpha_points[mut_alpha_mask])
        mut_points[mut_mask] = mut_points[mut_mask] @ rotation + transform

        # normalize both proteins
        points = np.concatenate([wt_points[wt_mask], mut_points[mut_mask]], axis=0)
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        max_l = (np.abs(points).max(axis=0)).max(axis=-1)
        wt_points[wt_mask] = (wt_points[wt_mask] - center) / max_l
        mut_points[mut_mask] = (mut_points[mut_mask] - center) / max_l

        wt_acids = np.array([self.acid_mapper[aa] for aa in row['protein_sequence']], dtype=np.int64)
        wt_acids = np.concatenate(
            [wt_acids, np.ones(self.seq_len - len(wt_acids), dtype=np.int64) * len(self.acid_mapper)])
        mut_acids = np.array([self.acid_mapper[aa] for aa in row['protein_sequence_mut']], dtype=np.int64)
        mut_acids = np.concatenate(
            [mut_acids, np.ones(self.seq_len - len(mut_acids), dtype=np.int64) * len(self.acid_mapper)])

        return {
            'wt_points': wt_points,
            'wt_mask': wt_mask,
            'wt_alpha_points': wt_alpha_points,
            'wt_alpha_mask': wt_alpha_mask,
            'wt_features': wt_features,
            'wt_atom_ids': wt_atom_ids,
            'wt_acids': wt_acids,
            'mut_points': mut_points,
            'mut_mask': mut_mask,
            'mut_features': mut_features,
            'mut_atom_ids': mut_atom_ids,
            'mut_alpha_points': mut_alpha_points,
            'mut_alpha_mask': mut_alpha_mask,
            'mut_acids': mut_acids,
            'dT': row['dT'].astype(np.float32),
            'pH': row['pH'].astype(np.float32)
        }

    def save_features(self, idx):
        features = self.calc_features(idx)
        for k in features.keys():
            if features[k].dtype == np.float32:
                features[k] = features[k].astype(np.float16)
            elif features[k].dtype == np.int64:
                features[k] = features[k].astype(np.int16)
        np.savez_compressed(f'{self.precompute}/{idx}', features)

    def load_features(self, idx):
        if self.precomputed is not None:
            true_idx = self.true_indexes[idx]
            features = np.load(f'{self.precomputed}/{true_idx}.npz', allow_pickle=True)['arr_0'].item()
            for k in features.keys():
                if features[k].dtype == np.float16:
                    features[k] = features[k].astype(np.float32)
                elif features[k].dtype == np.int16:
                    features[k] = features[k].astype(np.int64)
        else:
            features = self.calc_features(idx)
        return features

    def __iter__(self):
        return self

    def __next__(self):
        idx = random.randint(0, len(self.df) - 1)
        return self.load_features(idx)
