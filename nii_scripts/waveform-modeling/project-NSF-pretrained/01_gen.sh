#!/bin/sh

# generating
cd MODELS
if [ ! -e "./NSF" ];then
    tar -xzvf models.tar.gz
fi
cd ../


# Which pre-trained model?
# NSF: original NSF
# NSF-FFCond: NSF using a condition module with only one feedforward layer
# NSF-L3: NSF using only one spectral distance
# NSF-N2: without affine-transformation in neural fitler block
# NSF-S3: without sine-excitation

MODEL=NSF

# Directories of the input features, which are separated by ','
export TEMP_WAVEFORM_MODEL_INPUT_DIRS=$PWD/../TESTDATA-for-pretrained/mfbsp,$PWD/../TESTDATA-for-pretrained/f0

# Path to the model directory
export TEMP_WAVEFORM_MODEL_DIRECTORY=$PWD/MODELS/${MODEL}

# Path to the directory that will save the generated waveforms
export TEMP_WAVEFORM_OUTPUT_DIRECTORY=$PWD/MODELS/${MODEL}/output_trained_network

# Path to the trained_network.jsn (or epoch*.autosave)
export TEMP_WAVEFORM_MODEL_NETWORK_PATH=$PWD/MODELS/${MODEL}/trained_network.jsn

# Path to a temporary directory to save intermediate files (which will be deleted after generation)
export TEMP_WAVEFORM_TEMP_OUTPUT_DIRECTORY=$PWD/MODELS/${MODEL}/output_trained_tmp


python ../SCRIPTS/02_genwaveform.py config
