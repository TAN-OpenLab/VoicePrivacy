#!/bin/bash
. path.sh
. cmd.sh

rand_level="spk"
cross_gender="false"
distance="cosine"
proximity="farthest"

rand_seed=2020

stage=1

. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: "
  echo "  $0 [options] <src-data-dir> <pool-data-dir> <xvector-out-dir> <plda-dir>"
  echo "Options"
  echo "   --rand-level=utt     # [utt, spk] Level of randomness while computing the pseudo-xvectors"
  echo "   --rand-seed=<int>     #  Random seed while computing the pseudo-xvectors"
  echo "   --cross-gender=true     # [true, false] Whether to select same or
                                                   other gender while computing the pseudo-xvectors"
  exit 1;
fi

src_data=$1
#pool_data=$2
xvec_out_dir=$3
#plda_dir=$4

src_dataname=$(basename $src_data)
#pool_dataname=$(basename $pool_data)
src_xvec_dir=${xvec_out_dir}/xvectors_${src_dataname}
#pool_xvec_dir=${xvec_out_dir}/myXvectors_${pool_dataname}
#affinity_scores_dir=${src_xvec_dir}/spk_pool_scores
src_xvec_matrix_dir=${src_xvec_dir}/myMatrix_xvecs
pseudo_xvecs_dir=${src_xvec_dir}/myLshPseudo_xvecs
pseudo_One_xvecs_dir=${src_xvec_dir}/myLshPseudo_xvecs/EachOne_xvecs
mkdir -p ${affinity_scores_dir} ${pseudo_xvecs_dir} ${pseudo_One_xvecs_dir}


# Iterate over all the source speakers and generate 
# affinity distribution over anonymization pool
src_spk2gender=${src_data}/spk2gender
#pool_spk2gender=${pool_data}/spk2gender

if [ $stage -le 0 ]; then
  if [ "$distance" = "cosine" ]; then
    echo "Computing cosine similarity between source to each pool speaker."
    python local/anon/compute_spk_pool_cosine.py ${src_xvec_dir} ${pool_xvec_dir} \
	    ${affinity_scores_dir}
  elif [ "$distance" = "plda" ]; then
    echo "Computing PLDA affinity scores of each source speaker to each pool speaker."
    cut -d\  -f 1 ${src_spk2gender} | while read s; do
      #echo "Speaker: $s"
      local/anon/compute_spk_pool_affinity.sh ${plda_dir} ${src_xvec_dir} ${pool_xvec_dir} \
	   "$s" "${affinity_scores_dir}/affinity_${s}" || exit 1;
    done
  fi
fi

#为每个说话人设计x-vector矩阵
if [ $stage -le 1 ]; then
  python local/anon/gen_matrix_xvecs.py ${src_xvec_dir} ${src_xvec_matrix_dir} ${src_data} || exit 1;
fi



#将x-vector矩阵进行LSH，生成pseudo_xvecs
if [ $stage -le 2 ]; then
  python local/anon/gen_LSHpseudo_xvecs.py  ${src_xvec_matrix_dir} ${pseudo_xvecs_dir} ${src_data} ${pseudo_One_xvecs_dir} || exit 1;
fi

