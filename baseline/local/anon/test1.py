import sys
from os.path import basename, join
import operator

import numpy as np
import random
from kaldiio import WriteHelper, ReadHelper
from E2LSH import gen_e2LSH_family,gen_HashVals,H2


src_xvec_dir = "/root/Voice-Privacy-Challenge-2020/baseline/exp/models/2_xvect_extr/exp/xvector_nnet_1a/anon/xvectors_libri_dev_enrolls"
src_xvec_matrix_dir = "/root/Voice-Privacy-Challenge-2020/baseline/exp/models/2_xvect_extr/exp/xvector_nnet_1a/anon/xvectors_libri_dev_enrolls/myMatrixTest"
src_data = "/root/Voice-Privacy-Challenge-2020/baseline/data/libri_dev_enrolls"

src_spk2gender_file = join(src_data, 'spk2genderTest')
src_spk2utt_file = join(src_data, 'spk2uttTest')


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
src_xvec_scp = join(src_xvec_dir, 'xvectorTest.scp')
src_xvectors = {}
c = 0
#with open(pool_xvec_file) as f:
 #   for key, xvec in kaldi_io.read_vec_flt_scp(f):

with ReadHelper('scp:'+src_xvec_scp) as reader:
    for key, xvec in reader:
        src_xvectors[key] = xvec
        c += 1
# print("Read ", c, "src xvectors")


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
        print("+1") 
        break

pseudo_xvec_map = {}
#LSH begin

for key, mat in matrix_xvec_map.items() :
    C = pow(2, 32) - 5
    mat = mat.T
    print(mat[0])
    print(mat[1])
    dataFp = []
    n = len(mat[0]) #数据集列数 获得列表的列数,数据集传入5个向量,m=len(dataSet)获得行数
    m = len(mat)
    print(n)
    print(m)
    e2LSH_family = gen_e2LSH_family(n, 20, 512)  #参数512
    fpRand = [random.randint(-10, 10) for i in range(20)]
    for dataIndex in range(m):  # 对文档中所有向量进行运算
        # generate k hash values 对一个数据向量生成k个哈希值
        spkOneList =  mat[dataIndex].tolist()
        hashVals = gen_HashVals(e2LSH_family, spkOneList, 512)
        # generate fingerprint  生成指纹向量
        fp = H2(hashVals, fpRand, 20, C)
        dataFp.append(fp)
    dataFp = np.array(dataFp)
    print(dataFp)
    pseudo_xvec_map[key] = dataFp


for spk, gender in src_spk2gender.items():
    # Filter the affinity pool by gender
    matrix_gender_map[spk] = gender


# # Write features as ark,scp
# print("Writing matrix-speaker xvectors to: "+src_xvec_matrix_dir)
# ark_scp_output = 'ark,scp:{}/{}.ark,{}/{}.scp'.format(
#                     src_xvec_matrix_dir, 'matrix_xvector',
#                     src_xvec_matrix_dir, 'matrix_xvector')
# with WriteHelper(ark_scp_output) as writer:
#       for uttid, xvec in matrix_xvec_map.items():
#           writer(uttid, xvec)

# print("Writing matrix-speaker spk2gender.")
# with open(join(src_xvec_matrix_dir, 'spk2gender'), 'w') as f:
#     spk2gen_arr = [spk+' '+gender for spk, gender in matrix_gender_map.items()]
#     f.write('\n'.join(spk2gen_arr) + '\n')


