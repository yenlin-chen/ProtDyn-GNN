import requests
from os import path, makedirs
from shutil import copyfile
import numpy as np

def handle_timeout(signum, frame):
    raise TimeoutError

def fetch_ids_from_rcsb(payload, save_dir): #, verbose=True):

    '''
    Makes HTTP GET request to RCSB api for a list of PDB-chain IDs.
    '''

    url_rcsb = 'https://search.rcsb.org/rcsbsearch/v1/query?json='

    print('Making GET request to RCSB...')
    returned_data = requests.get(url_rcsb+payload)

    status_code = returned_data.status_code
    if status_code == 200:
        print('HTTP GET request to RCSB successful')
    else:
        raise ConnectionError(f'HTTP GET request to RCSB responded with '
                              f'status code {status_code}')

    # decode returned data into dictionary
    decoded_data = returned_data.json()

    # list of entities (pdb-entity)
    entities = [entry['identifier'].replace('_', '-')
                    for entry in decoded_data['result_set']]

    # print summary progress
    print(f' -> {decoded_data["group_by_count"]} out of total '
          f'{decoded_data["total_count"]} proteins recieved')

    if save_dir is not None:
        makedirs(save_dir, exist_ok=True)
        print(f'Saving...')
        np.savetxt(path.join(save_dir, default_entity_filename),
                   entities, fmt='%s')

    print(f' -> {len(entities)} entites')
    print('Done')
    return entities

def write_to_file(msg, f_handle):
    if f_handle is not None:
        f_handle.write(f'{msg}\n')
        f_handle.flush()

def append_to_file(msg, filename):
    with open(filename, 'a') as f_out:
        f_out.write(msg+'\n')
        f_out.flush()

def vprint(verbose, *args, **kargs):
    if not isinstance(verbose, bool):
        raise ValueError('Please specify verbosity for vprint()')
    if verbose:
        print(*args, **kargs)

def id_to_filename(id):
    if len(id) == 4:
        return id.upper()
    else:
        return f'{id[:4].upper()}-{id[5:]}'

def backup_file(filename):
    if path.exists(filename):
        idx = 1
        loc_ext = filename.rfind('.')
        while path.exists(filename[:loc_ext]+f'-{idx}'+filename[loc_ext:]):
            idx += 1
        copyfile(filename, filename[:loc_ext]+f'-{idx}'+filename[loc_ext:])

def rm_log_entries(filename, id_list):
    if path.exists(filename):
        log_content, logged_ids = read_logs(filename)
        new_log_content = []
        for idx, ID in enumerate(logged_ids):
            if ID not in id_list:
                new_log_content.append(log_content[idx])
        open(filename, 'w').close()
        with open(filename, 'w+') as f_out:
            for line in new_log_content:
                f_out.write(line+'\n')

def read_logs(*logfiles, backup=False):
    log_content = []
    logged_entries = []

    for logfile in logfiles:
        if path.exists(logfile):
            with open(logfile, 'r') as f_in:
                log_content += f_in.read().splitlines()

            # save a backup copy
            if backup:
                backup_file(logfile)

    logged_entries += [line.split(' ')[0] for line in log_content]

    return log_content, logged_entries

def check_id_type(id_list):
    if isinstance(id_list, (list, np.ndarray)):
        # if all entries have more than 5 characters
        if all( (len(ID)>5 for ID in id_list) ):
            entry_type = 'chain'
        elif all( (len(ID)==4 for ID in id_list) ):
            entry_type = 'pdb'
        else:
            entry_type = 'mixed'

    elif isinstance(id_list, str):
        if len(id_list) > 5:
            entry_type = 'chain'
        elif len(id_list) == 4:
            entry_type = 'pdb'
        else:
            entry_type = 'unknown'

    else:
        raise ValueError('Function can only check PDB entry or chain IDs')

    return entry_type
