'''
After MF-GO is downloaded for all similarity datasets, re-do the
conversion process to ensure that every dataset is built upon the same
pool of downloaded MF-GO (to prevent the case where some MF-GO that
failed to download was retrieved successfully in subsequent query
processes).
'''

import numpy as np
from os import path, makedirs
from tqdm import tqdm
import requests
import json

entities = np.loadtxt('pdb_entity.txt', dtype=np.str_)

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
all_pdbs = np.loadtxt(path.join(cache_dir, '..', 'all_pdbs.txt'),
                      dtype=np.str_)
for idx, entity in enumerate(tqdm(entities,
                                   ascii=True,
                                   dynamic_ncols=True)):
    pdbID, entityID = entity.split('_')
    if pdbID not in all_pdbs:
        continue # only process those with mfgo downloaded

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
