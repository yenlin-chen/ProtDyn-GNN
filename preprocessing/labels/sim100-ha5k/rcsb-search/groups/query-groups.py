'''
GET request using RCSB Search API.
'''

import json
import requests
import numpy as np
from tqdm import tqdm
from os import path, makedirs

########################################################################
# query RCSB to get a list of entities
########################################################################

# payload_file = 'payload.json'

# # get payload text from file
# with open(payload_file, 'r') as f:
#     payload = f.read()

# # GET request
# url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={payload}"
# data = requests.get(url)

# # decode returned data if request is successful
# if data.status_code != 200:
#     with open('error.txt', 'w+') as f:
#         f.write(f'{data.text}')
#         f.flush()
#     raise RuntimeError(f' -> GET Request failed with code {data.status_code}')
# decoded = data.json()

# with open('pdb_entity-groups.csv', 'w+') as f:
#     json.dump(decoded, f,
#               indent=2, separators=(',', ': '))

########################################################################
# SORT IT OUT
########################################################################

with open('pdb_entity-groups.csv', 'r') as f:
    decoded = json.load(f)

set_of_groups = []
len_sets = []
for group in decoded['group_set']:
    group_list = [item['identifier'][:4] for item in group['result_set']]

    set_of_groups.append(group_list)
    len_sets.append(len(group_list))

uni, cnt = np.unique(len_sets, return_counts=True)
print(np.vstack((uni, cnt)).T)
# np.savetxt('counts.csv', np.vstack((uni, cnt)).T,
#            header='group_size number_of_groups_with_this_size',
#            fmt='%d %d')
