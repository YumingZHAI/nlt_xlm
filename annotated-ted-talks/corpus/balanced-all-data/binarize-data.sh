#!/usr/bin/env bash

WORK_PATH="/Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master"
TOOLS_PATH=$WORK_PATH/tools

FASTBPE=$TOOLS_PATH/fastBPE/fast
CODES_PATH=$WORK_PATH/codes_xnli_15
VOCAB_PATH=$WORK_PATH/vocab_xnli_15

for side in en fr; do
    # preprocess.py: data = Dictionary.index_data(txt_path, bin_path, dico)
    python3 $WORK_PATH/preprocess.py ${VOCAB_PATH} balanced.${side}.bpe
done


