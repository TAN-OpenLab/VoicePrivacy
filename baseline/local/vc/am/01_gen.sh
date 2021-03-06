#!/bin/sh
# -------
# input feature directories
#  here, we use features in ../TESTDATA/vctk_vctk_anonymize for demonstration
# 
. path.sh
. local/vc/am/init.sh

proj_dir=${nii_scripts}/acoustic-modeling/project-DAR-continuous
test_data_dir=$1

output_dir=${test_data_dir}/am_out_mel
output_tmp_dir=${test_data_dir}/am_out_tmp
export TEMP_ACOUSTIC_MODEL_INPUT_DIRS=${test_data_dir}/ppg,${test_data_dir}/xvector,${test_data_dir}/f0
# where is the directory of the trained model
export TEMP_ACOUSTIC_MODEL_DIRECTORY=exp/models/3_ss_am
# where is the trained model?
#  here, we use network.jsn for demonstration.
#  of course, it will generate random noise only
export TEMP_ACOUSTIC_NETWORK_PATH=${TEMP_ACOUSTIC_MODEL_DIRECTORY}/trained_network.jsn
# where to store the features generated by the trained network?
export TEMP_ACOUSTIC_OUTPUT_DIRECTORY=${output_dir}
# directory to save intermediate files (it will be deleted after)
export TEMP_ACOUSTIC_TEMP_OUTPUT_DIRECTORY=${output_tmp_dir}

temp_dir="exp/tmp"
mkdir -p $temp_dir
export TEMP_ADDITIONAL_COMMAND="--cache_path $temp_dir"

# 
python ${proj_dir}/../SCRIPTS/03_syn.py config_libri_am || exit 1
# after running this scripts, the generated features should be in ${TEMP_ACOUSTIC_OUTPUT_DIRECTORY}

rm -r ${TEMP_ACOUSTIC_TEMP_OUTPUT_DIRECTORY}
