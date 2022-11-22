import hashlib
import os
import os

import pandas as pd
import requests
from tqdm import tqdm


def get_hash(x):
    return hashlib.sha1(x.encode()).hexdigest()


root = '/home/sthfaceless'

df = pd.read_csv(f'{root}/novozymes/train_with_groups.csv')
# filter only grouped
df = df.loc[df.group != -1]
counts = df.group.value_counts()
group_counts = dict(zip(counts.index.tolist(), counts.tolist()))
df = df[df.group.apply(lambda val: group_counts[val] > 1)]
# filter only lower than 400
df['seq_len'] = df['protein_sequence'].apply(len)
df = df[(df['seq_len'] <= 400)]
# filter already downloaded
os.makedirs(f'{root}/pdb', exist_ok=True)
items_downloaded = set([path.split('.')[0] for path in os.listdir(f'{root}/pdb')])
df['seq_hash'] = df['protein_sequence'].apply(lambda seq: get_hash(seq)[:6])
df = df[df['seq_hash'].apply(lambda hash: hash not in items_downloaded)]
print(df.describe())

seqs = df['protein_sequence'].tolist()
for seq in tqdm(seqs, desc='Sequences downloaded'):
    esmfold_api_url = 'https://api.esmatlas.com/foldSequence/v1/pdb/'
    while True:
        r = requests.post(esmfold_api_url, data=seq[:400])
        if r.status_code == 200:
            structure = r.text
            seq_id = get_hash(seq)[:6]
            with open(f'{root}/pdb/{seq_id}.pdb', 'w') as f:
                f.write(structure)
            break
        else:
            print(r.status_code)
