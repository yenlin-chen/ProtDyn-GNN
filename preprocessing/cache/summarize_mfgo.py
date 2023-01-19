import json
from os import listdir
from numpy import savetxt

mfgo_dir = './mfgo/'
keys = ['category', 'definition', 'identifier', 'name']

all_pdbs = []
all_mfgo = {}
for json_file in [f for f in listdir(mfgo_dir) if '.json' in f]:
    all_pdbs.append(json_file.replace('.json', ''))
    with open(mfgo_dir+json_file, 'r') as f:
        go_dict = json.load(f)

    for code in go_dict:
        if go_dict[code]['category'] == 'Molecular_function':
            if code not in all_mfgo.keys():
                all_mfgo[code] = {key: go_dict[code][key] for key in keys}

with open('./non-exclusive_mfgo_list.json', 'w+') as f:
    json.dump(all_mfgo, f,
              indent=4, separators=(',', ': '), sort_keys=True)

savetxt('./non-exclusive_pdb_list.txt', all_pdbs, fmt='%s')
