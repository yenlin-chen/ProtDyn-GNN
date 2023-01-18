import json
import numpy as np

# load all mfgo
all_mfgo_file = '../../../../cache/non-exclusive_mfgo_list.json'
with open(all_mfgo_file, 'r') as f:
    all_mfgo = json.load(f)

# load count file
cnt_file = './mfgo-count.txt'
data = np.loadtxt(cnt_file, dtype=np.str_)

# get sorted list of mfgo
code = data[:,0]
cnt = data[:,1].astype(np.int_)
idx = np.flip(np.argsort(cnt))

mfgo_sorted = {}
names_sorted = []
for i in idx:
    names_sorted.append(all_mfgo[code[i]]['name'])
    mfgo_sorted[code[i]] = all_mfgo[code[i]]


# save data to drive
np.savetxt('./mfgo-count-sorted.txt',
           np.column_stack((idx, code[idx], cnt[idx], names_sorted)),
           header='id annotation count name',
           fmt='%-4s %-10s %-4s %s')
with open('./top-mfgos.json', 'w+') as f:
    json.dump(mfgo_sorted, f,
              indent=4, separators=(',', ': '), sort_keys=False)
