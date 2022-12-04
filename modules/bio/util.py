import hashlib
import os

import pandas as pd
from Bio.SVDSuperimposer import SVDSuperimposer
from biopandas.pdb import PandasPdb

from modules.common.util import *


def get_hash(x):
    return hashlib.sha1(x.encode()).hexdigest()


def agg_ph(df, base_ph=8.0):
    best_idx, best_dist = 0, 1e6
    for idx in range(len(df)):
        dist = abs(df.iloc[idx]['pH'] - base_ph)
        if dist < best_dist:
            best_idx = idx
            best_dist = dist

    return df.iloc[best_idx]


def agg_pairs(df, train=False):
    rows = {'protein_sequence': [], 'protein_sequence_mut': [],
            'ddG': None, 'dT': [], 'pdb_path': [], 'mut_path': [], 'pH': [],
            'group': [], 'seq_id': [], 'mut_id': []}

    for first_idx in range(len(df)):
        if not train and first_idx != 0:
            break
        for second_idx in range(first_idx + 1, len(df)):
            first_row = df.iloc[first_idx]
            second_row = df.iloc[second_idx]
            rows['protein_sequence'].append(first_row['protein_sequence'])
            rows['protein_sequence_mut'].append(second_row['protein_sequence'])
            if train:
                rows['dT'].append(second_row['tm'] - first_row['tm'])
            rows['pdb_path'].append(first_row['pdb_path'])
            rows['mut_path'].append(second_row['pdb_path'])
            rows['pH'].append((first_row['pH'] + second_row['pH']) / 2)
            rows['group'].append(first_row['group'])
            rows['seq_id'].append(first_row['seq_id'])
            rows['mut_id'].append(second_row['seq_id'])

    if not train:
        del rows['dT']
    return pd.DataFrame(rows)


def make_mutation_df(df, pdb_root, train=True):
    df = df.loc[df.group != -1]
    # ESMFold prediction accurate only for <= 400 seq lengths
    df = df[df['protein_sequence'].apply(len) <= 400]

    # left proteins with predicted ESMFold pdb files

    df['pdb_path'] = df['protein_sequence'].apply(lambda seq: f'{get_hash(seq)[:6]}.pdb')
    pdbs_paths = set(os.listdir(pdb_root))
    df = df[df['pdb_path'].apply(lambda x: os.path.basename(x) in pdbs_paths)]

    # when sequence is same left proteins with pH similar to test pH
    df = df.groupby('protein_sequence', as_index=False, sort=False).apply(agg_ph)

    # remove left groups with 1 element
    counts = df.group.value_counts()
    group_counts = dict(zip(counts.index.tolist(), counts.tolist()))
    df = df[df.group.apply(lambda val: group_counts[val] > 1)]

    # make wt - mutant pairs
    df = df.groupby('group', as_index=False, sort=False).apply(lambda aggdf: agg_pairs(aggdf, train)).reset_index(
        drop=True)
    return df


def process_sequence(seq):
    #     return seq.translate({ord(c): None for c in 'X'})
    return seq


def process_extra_df_row(row, extra_pdb_root):
    # add mutant sequence row
    mut_seq, mut_pos = row['sequence'], row['seq_position']
    mut_seq = mut_seq[:mut_pos] + row['mutant'] + mut_seq[mut_pos + 1:]
    row['protein_sequence_mut'] = process_sequence(mut_seq)
    row['protein_sequence'] = process_sequence(row['sequence'])

    pdb_id = row['PDB_chain']
    row['pdb_path'] = f'{extra_pdb_root}/{pdb_id}/{pdb_id}_relaxed.pdb'

    pdb_pos = row['pdb_position']
    wt, mut = row['wildtype'], row['mutant']
    row['mut_path'] = f'{extra_pdb_root}/{pdb_id}/{pdb_id}_{wt}{pdb_pos}{mut}_relaxed.pdb'
    return row


def load_ddg_data(path, extra_pdb_root):
    ddgdf = pd.read_csv(path)
    ddgdf = ddgdf.apply(lambda row: process_extra_df_row(row, extra_pdb_root), axis=1) \
        .drop(['wildtype', 'pdb_position', 'mutant', 'wT', 'source', 'PDB_chain', 'seq_position', 'sequence'], axis=1)
    ddgdf = ddgdf[
        ddgdf['pdb_path'].apply(lambda p: os.path.exists(p)) & ddgdf.mut_path.apply(lambda p: os.path.exists(p))]
    ddgdf = ddgdf[ddgdf['protein_sequence'] != ddgdf['protein_sequence_mut']]
    return ddgdf


def get_pdb_features(pdb_path, atoms_mapper, max_atoms, max_len, pe_features=16, protein_scale=100):
    atoms = PandasPdb().read_pdb(pdb_path).df['ATOM']

    # true points mask
    mask = np.zeros(max_atoms, dtype=bool)
    mask[np.arange(max_atoms) < len(atoms)] = True

    # atoms coordinates
    points = np.stack([atoms['x_coord'], atoms['y_coord'], atoms['z_coord']], axis=-1).astype(np.float32)
    points -= points[0][None, :]
    points /= 2 * protein_scale

    # get coordinates of alpha atoms for superimposing
    alpha_points = points[atoms['atom_name'].apply(lambda name: name in ('C'))]
    # true alphas mask
    alpha_mask = np.zeros(max_len, dtype=bool)
    alpha_mask[np.arange(max_len) < len(alpha_points)] = True

    # atoms ids for embedding
    atom_ids = [atoms_mapper[atom] for atom in atoms['atom_name'].tolist()]
    atom_ids = np.array(atom_ids, dtype=np.int64)

    # some features for atoms
    pos_features = get_numpy_positional_encoding(points, pe_features)
    features = np.concatenate([pos_features], axis=-1)

    # pad all vectors to max_atoms
    points = np.concatenate([points, np.zeros((max_atoms - len(points), 3), dtype=np.float32)], axis=0)
    atom_ids = np.concatenate([atom_ids, np.ones((max_atoms - len(atom_ids)), dtype=np.int64) * len(atoms_mapper)],
                              axis=0)
    features = np.concatenate([features, np.zeros((max_atoms - len(features), features.shape[-1]), dtype=np.float32)],
                              axis=0)
    alpha_points = np.concatenate([alpha_points, np.zeros((max_len - len(alpha_points), 3), dtype=np.float32)], axis=0)
    return points, features, atom_ids, mask, alpha_points, alpha_mask


def get_protein_transform(x, y):
    sup = SVDSuperimposer()
    min_len = min(len(x), len(y))
    sup.set(x[:min_len], y[:min_len])
    sup.run()
    rot, tran = sup.get_rotran()
    return rot, tran



