import json
from os import path

self_dir = path.dirname(path.realpath(__file__))

# from 3-letter symbols to indices
with open(path.join(self_dir, 'res_code3-to-index.json'), 'r') as fin:
    res3_dict = json.load(fin)

# from 1-letter symbols to indices
with open(path.join(self_dir, 'res_code1-to-index.json'), 'r') as fin:
    res1_dict = json.load(fin)
