'''
Converts list of PDB-entity ID into PDB-chain ID.
Entities without annotations are discardeds.
'''

import numpy as np
from os import path, chdir, getcwd, makedirs
from tqdm import tqdm
import json
import requests

# define directories
cwd = getcwd()
cache_dir = '../../../cache/mfgo'

# define url
url = 'https://www.ebi.ac.uk/pdbe/api/mappings/go/'

# read PDB-entity IDs
entities = np.loadtxt('pdb_entity.txt', dtype=np.str_)

# open file resources
err = open('go_download_failed.txt', 'w+')
success = open('pdb_chain.txt', 'w+')
id_list_file = open('../target/id_list.txt', 'w+')
makedirs('../target', exist_ok=True)

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