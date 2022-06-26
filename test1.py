
import sys
import numpy as np
sys.path.append("/root/Voice-Privacy-Challenge-2020/nii/pyTools/")
from os.path import join, basename
from ioTools import readwrite
from kaldiio import WriteHelper, ReadHelper

data_dir = "/root/Voice-Privacy-Challenge-2020/baseline/data/libri_dev_enrolls"
xvector_file = "../../exp/models/2_xvect_extr/exp/xvector_nnet_1a/anon/xvectors_libri_dev_enrolls/pseudo_xvecs/test1.scp"
out_dir ="/root/Voice-Privacy-Challenge-2020/baseline/local/featex/test"

dataname = basename(data_dir)
yaap_pitch_dir = join(data_dir, 'yaapt_pitch')
xvec_out_dir = join(out_dir, "testXvector")
pitch_out_dir = join(out_dir, "testF0")
pitch_file = join(data_dir, 'pitch.scp')
pitch2shape = {}

with ReadHelper('scp:'+pitch_file) as reader:
    for key, mat in reader:
        print(key,mat.shape)
        k = mat[:,1]
        kaldi_f0 = mat[:,1].squeeze().copy()
        yaapt_f0 = readwrite.read_raw_mat(join(yaap_pitch_dir, key+'.f0'), 1)
        print(yaapt_f0)
        print('\n')
        f0 = np.zeros(kaldi_f0.shape)
        f0[:yaapt_f0.shape[0]] = yaapt_f0
        print(f0.shape)
        break

with ReadHelper('scp:'+xvector_file) as reader:
    for key, mat in reader:
        print(key)
        print(mat)
        print(mat.shape)
        plen = 584
        mat = mat[np.newaxis]
        print(plen)
        print(mat)
        xvec = np.repeat(mat, plen, axis=0)
        print(xvec)
        print(xvec.shape)
        break
