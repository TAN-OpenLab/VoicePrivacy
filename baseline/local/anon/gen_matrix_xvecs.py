import sys
from os.path import basename, join
import operator

import numpy as np
import random
from kaldiio import WriteHelper, ReadHelper

args = sys.argv
print(args)

src_xvec_dir = args[1]
src_xvec_matrix_dir = args[2]
src_data = args[3]

src_spk2gender_file = join(src_data, 'spk2gender')
src_spk2utt_file = join(src_data, 'spk2utt')


src_spk2gender = {}
src_spk2utt = {}
src_spk2utt_num = {}
# Read source spk2gender and spk2utt
print("Reading source spk2gender.")
with open(src_spk2gender_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        src_spk2gender[sp[0]] = sp[1]


matrix_xvec_map = {}
matrix_gender_map = {}
src_xvec_scp = join(src_xvec_dir, 'xvector.scp')
src_xvectors = {}
c = 0


with ReadHelper('scp:'+src_xvec_scp) as reader:
    for key, xvec in reader:
        src_xvectors[key] = xvec
        c += 1
print("Read ", c, "src xvectors")



print("Reading source spk2utt.")
with open(src_spk2utt_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        src_spk2utt[sp[0]] = sp[1:]
        src_spk2utt_num[sp[0]] = len(sp[1:])
        matrix_spk_matrix = np.zeros((len(sp[1:]), 512), dtype='float64')
        for i, key in enumerate(sp[1:]):
            matrix_spk_matrix[i, :] = src_xvectors[key]
        matrix_xvec_map[sp[0]] = matrix_spk_matrix


for spk, gender in src_spk2gender.items():
    # Filter the affinity pool by gender
    matrix_gender_map[spk] = gender


# Write features as ark,scp
print("Writing matrix-speaker xvectors to: "+src_xvec_matrix_dir)
ark_scp_output = 'ark,scp:{}/{}.ark,{}/{}.scp'.format(
                    src_xvec_matrix_dir, 'matrix_xvector',
                    src_xvec_matrix_dir, 'matrix_xvector')
with WriteHelper(ark_scp_output) as writer:
      for uttid, xvec in matrix_xvec_map.items():
          writer(uttid, xvec)

print("Writing matrix-speaker spk2gender.")
with open(join(src_xvec_matrix_dir, 'spk2gender'), 'w') as f:
    spk2gen_arr = [spk+' '+gender for spk, gender in matrix_gender_map.items()]
    f.write('\n'.join(spk2gen_arr) + '\n')


