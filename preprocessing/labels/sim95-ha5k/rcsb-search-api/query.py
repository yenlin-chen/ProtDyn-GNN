'''
GET request using RCSB Search API to download pdbID-entityID.
'''

import json
import requests
import numpy as np
from tqdm import tqdm
from os import path, makedirs

########################################################################
# query RCSB to get a list of entities
########################################################################

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

print(f" -> {decoded['total_count']} entities received from RCSB")
print(f" -> {len(decoded['result_set'])} entities after applying filter")

# convert decoded data into lists
entities = [entry['identifier'] for entry in decoded['result_set']]
pdbs = np.unique([ID[:4] for ID in entities])

# save list to corresponding directory
np.savetxt('pdb_entity.txt', entities, fmt='%s')
np.savetxt('pdb.txt', pdbs, fmt='%s')

########################################################################
# convert entity to chains
########################################################################

# define directories
cache_dir = path.join('..', '..', '..', 'cache', 'mfgo')
target_dir = 'target-copy_to_upper_level'

# define url
url = 'https://www.ebi.ac.uk/pdbe/api/mappings/go/'

# open file resources
makedirs(target_dir, exist_ok=True)
err = open('go_download_failed.txt', 'w+')
success = open('pdb_chain.txt', 'w+')
id_list_file = open(path.join(target_dir, 'id_list-auth_asym_id.txt'), 'w+')

# download MF-GO annotations and convert entity ID to chain ID
for idx, entity in enumerate(tqdm(entities,
                                   ascii=True,
                                   dynamic_ncols=True)):
    pdbID, entityID = entity.split('_')
    mfgo_cache = path.join(cache_dir, f'{pdbID}.json')

    # check if cache is on disk
    if path.exists(mfgo_cache):
        with open(mfgo_cache, 'r') as f:
            go_dict = json.load(f)
    else:
        # try to download annotations
        try:
            data = requests.get(url+pdbID, timeout=10)
        except requests.Timeout:
            msg = f'{pdbID} -> GET request timeout'
            err.write(msg+'\n')
            err.flush()
            # tqdm.write(msg)
            continue

        # failure on the server side
        if data.status_code != 200:
            msg = f'{pdbID} -> GET request failed with code {data.status_code}'
            err.write(msg+'\n')
            err.flush()
            # tqdm.write(msg)
            continue

        decoded = data.json()
        go_dict = decoded[pdbID.lower()]['GO']

        # save cache if download is new
        with open(mfgo_cache, 'w+') as f_out:
            json.dump(go_dict, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

    mfgo = {}
    for code in go_dict:
        if go_dict[code]['category'] == 'Molecular_function':
            mfgo[code] = {'category': 'Molecular_function',
                          'mappings': go_dict[code]['mappings']}

    # get all chains in entity
    chainIDs = []
    for code in mfgo:
        for m in mfgo[code]['mappings']:
            if int(m['entity_id']) == int(entityID):
                chainIDs.append(m['chain_id'])

    # remove duplicate chains
    chainIDs = np.unique(chainIDs)

    for chainID in chainIDs:
        success.write(f'{pdbID}-{chainID}\n')
    success.flush()

    if chainIDs.size > 0:
        id_list_file.write(f'{pdbID}-{chainIDs[0]}\n')
        id_list_file.flush()

err.close()
success.close()
id_list_file.close()
