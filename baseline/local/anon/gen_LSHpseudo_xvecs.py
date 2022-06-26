import sys
from os.path import basename, join
import operator

import numpy as np
import random
from kaldiio import WriteHelper, ReadHelper
from E2LSH import gen_e2LSH_family,gen_HashVals,H2
from sklearn import preprocessing

args = sys.argv
print(args)

src_xvec_matrix_dir = args[1]
pseudo_xvecs_dir = args[2]
src_data = args[3]
pseudo_One_xvecs_dir = args[4]

matrix_spk2gender_file = join(src_xvec_matrix_dir, 'spk2gender')
src_spk2utt_file = join(src_data, 'spk2utt')

src_spk2gender = {}
src_spk2utt = {}
# Read source spk2gender and spk2utt
print("Reading source spk2gender.")
with open(matrix_spk2gender_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        src_spk2gender[sp[0]] = sp[1]

pseudo_gender_map = {}
for spk, gender in src_spk2gender.items():
    pseudo_gender_map[spk] = gender

print("Reading source spk2utt.")
with open(src_spk2utt_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        src_spk2utt[sp[0]] = sp[1:]

matrix_xvec_file = join(src_xvec_matrix_dir, 'matrix_xvector.scp')
matrix_xvectors = {}

c = 0
with ReadHelper('scp:'+matrix_xvec_file) as reader:
    for key, xvec in reader:
        matrix_xvectors[key] = xvec
        c += 1
print("Read ", c, "matrix xvectors")

#矩阵转置
# matrix_xvectors_T = {}
# for key, mat in matrix_xvectors.items():
#     matrix_xvectors_T[key] = mat.T



pseudo_xvec_map = {}
#LSH begin
for key, mat in matrix_xvectors.items() :
    C = pow(2, 32) - 5
    mat = mat.T
    dataFp = []
    n = len(mat[0]) #数据集列数 获得列表的列数,数据集传入5个向量,m=len(dataSet)获得行数
    m = len(mat)
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
    
    pseudo_xvec_map[key] = preprocessing.scale(dataFp)
#LSH end

Each_One_xvec_map = {}
#每个语音向量所对应的匿名化之后的x-vector begin
for spk, utt in src_spk2utt.items():
    for uttid in src_spk2utt[spk]:
            Each_One_xvec_map[uttid] = pseudo_xvec_map[spk]
#end

print("Writing pseud-speaker Each One xvectors to: "+pseudo_One_xvecs_dir)
ark_scp_output = 'ark,scp:{}/{}.ark,{}/{}.scp'.format(
                    pseudo_One_xvecs_dir, 'pseudo_EachOne_xvector',
                    pseudo_One_xvecs_dir, 'pseudo_EachOne_xvector')
with WriteHelper(ark_scp_output) as writer:
      for uttid, xvec in Each_One_xvec_map.items():
          writer(uttid, xvec)

# Write features as ark,scp
print("Writing pseud-speaker to: "+pseudo_xvecs_dir)
ark_scp_output = 'ark,scp:{}/{}.ark,{}/{}.scp'.format(
                    pseudo_xvecs_dir, 'pseudo_xvector',
                    pseudo_xvecs_dir, 'pseudo_xvector')
with WriteHelper(ark_scp_output) as writer:
      for spk, xvec in pseudo_xvec_map.items():
          writer(uttid, xvec)

print("Writing pseudo-speaker spk2gender.")
with open(join(pseudo_xvecs_dir, 'spk2gender'), 'w') as f:
    spk2gen_arr = [spk+' '+gender for spk, gender in pseudo_gender_map.items()]
    sorted_spk2gen = sorted(spk2gen_arr)
    f.write('\n'.join(sorted_spk2gen) + '\n')




