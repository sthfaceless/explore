import hashlib, os
import os

import pandas as pd
import requests
from tqdm import tqdm


def get_hash(x):
    return hashlib.sha1(x.encode()).hexdigest()

root = '/home/sthfaceless'

df = pd.read_csv(f'{root}/novozymes/train.csv')
print(df.describe())

os.makedirs(f'{root}/pdb', exist_ok=True)
items_downloaded = set([name.split('.')[0] for name in os.listdir(f'{root}/pdb')])

seqs = df['protein_sequence'].tolist()
for seq in tqdm(seqs):
    esmfold_api_url = 'https://api.esmatlas.com/foldSequence/v1/pdb/'
    seq_id = get_hash(seq)[:6]
    if seq_id in items_downloaded:
        continue
    while True:
        r = requests.post(esmfold_api_url, data=seq[:400])
        if r.status_code == 200:
            structure = r.text
            with open(f'{root}/pdb/{seq_id}.pdb', 'w') as f:
                f.write(structure)
            break
        else:
            print(r.status_code)
