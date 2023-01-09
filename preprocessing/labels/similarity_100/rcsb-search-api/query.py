'''
GET request using RCSB Search API to download pdbID-entityID.
'''

import requests
from tqdm import tqdm
import numpy as np
from os import makedirs

payload_file = 'payload.json'

# get payload text from file
with open(payload_file, 'r') as f:
    payload = f.read()

# GET request
url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={payload}"
data = requests.get(url)

# decode returned data if request is successful
if data.status_code != 200:
    with open('error.txt', 'w+') as f:
        f.write(f'{data.text}')
        f.flush()
    raise RuntimeError(f' -> GET Request failed with code {data.status_code}')
decoded = data.json()

print(f" -> {decoded['total_count']} proteins received from RCSB")

# convert decoded data into lists
polymers = [entry['identifier'] for entry in decoded['result_set']]
pdbs = np.unique([ID[:4] for ID in polymers])

# save list to corresponding directory
np.savetxt('pdb_entity.txt', polymers, fmt='%s')
np.savetxt('pdb.txt', pdbs, fmt='%s')
