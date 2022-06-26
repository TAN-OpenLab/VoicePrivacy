#!/bin/sh

# PATH to the pyTools
export TEMP_CURRENNT_PROJECT_PYTOOLS_PATH=/home/smg/wang/WORK/WORK/CODE/git_local/team/project-CURRENNT-public/pyTools

# PATH to currennt
export TEMP_CURRENNT_PROJECT_CURRENNT_PATH=/home/smg/wang/WORK/WORK/TOOL/local/bin/currennt

# PATH to SOX (http://sox.sourceforge.net/sox.html)
export TEMP_CURRENNT_PROJECT_SOX_PATH=/usr/bin/sox

# PATH to SV56 (a software to normalize waveform amplitude.
#  https://www.itu.int/rec/T-REC-P.56 (document)
#  https://www.itu.int/rec/T-REC-G.191-201901-I/en (code)
#  This software is not necessary, I used it because it is available in our lab.
#  You can use other tools to normalize the waveforms before put them into this project.
#  Then, you can set TEMP_CURRENNT_PROJECT_SV56_PATH=None)
export TEMP_CURRENNT_PROJECT_SV56_PATH=/home/smg/wang/WORK/WORK/TOOL/local/bin/sv56demo

# Add pyTools to PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:${TEMP_CURRENNT_PROJECT_PYTOOLS_PATH}
