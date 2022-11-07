import hashlib, os
import os

import pandas as pd
import requests
from tqdm import tqdm


def get_hash(x):
    return hashlib.sha1(x.encode()).hexdigest()


df = pd.read_csv('/home/sthfaceless/novozymes/train.csv')
print(df.describe())

os.makedirs('/home/sthfaceless/pdb', exist_ok=True)
items_downloaded = len(os.listdir('/home/sthfaceless/pdb'))

seqs = df['protein_sequence'].tolist()[items_downloaded:]
for seq in tqdm(seqs):
    esmfold_api_url = 'https://api.esmatlas.com/foldSequence/v1/pdb/'

    while True:
        r = requests.post(esmfold_api_url, data=seq[:400])
        if r.status_code == 200:
            structure = r.text
            seq_id = get_hash(seq)[:6]
            with open(f'/home/sthfaceless/pdb/{seq_id}.pdb', 'w') as f:
                f.write(structure)
            break
        else:
            print(r.status_code)
