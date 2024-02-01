# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# This script is meant to prepare data to reproduce XNLI experiments
# Just modify the "code" and "vocab" path for your own model
# This script will apply BPE using the XNLI15 bpe codes, and binarize data.

set -e

pair=$1  # input language pair

# data paths
MAIN_PATH=$PWD
PARA_PATH=$PWD/data/para
TOOLS_PATH=$PWD/tools
WIKI_PATH=$PWD/data/wiki
XNLI_PATH=$PWD/data/xnli/XNLI-1.0
PROCESSED_PATH=$PWD/data/processed/XLM15
CODES_PATH=$MAIN_PATH/codes_xnli_15
VOCAB_PATH=$MAIN_PATH/vocab_xnli_15
FASTBPE=$TOOLS_PATH/fastBPE/fast

# Get BPE codes and vocab  (I've already downloaded them)
#wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_15 -P $MAIN_PATH
#wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15 -P $MAIN_PATH

##### begin: prepare data to pretrain a language model (with MLM and TLM)
# first, download / preprocess monolingual data for the 15 languages
# Downloading the Wikipedia dumps make take several hours. The get-data-wiki.sh script will automatically
# download Wikipedia dumps, extract raw sentences, clean and tokenize them.
#for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
#  ./get-data-wiki.sh $lg
#done

#This script will download and tokenize the parallel data [used for the TLM objective]:
#lg_pairs="ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh"
#for lg_pair in $lg_pairs; do
#  ./get-data-para.sh $lg_pair
#done

## Prepare monolingual data
# apply BPE codes and (use the dictionary to) binarize the monolingual corpora
# cf https://github.com/facebookresearch/XLM#train-your-own-monolingual-bert-model
#for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
#    for split in train valid test; do
#    #        applybpe output input codes [vocab]  => apply BPE codes to a text file
#    $FASTBPE applybpe $PROCESSED_PATH/$lg.$split $WIKI_PATH/txt/$lg.$split $CODES_PATH
#    python preprocess.py $VOCAB_PATH $PROCESSED_PATH/$lg.$split
#    done
#done
#
### Prepare parallel data
## apply BPE codes and binarize the parallel corpora
#for pair in ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh; do
#    for lg in $(echo $pair | sed -e 's/\-/ /g'); do
#        for split in train valid test; do
#            $FASTBPE applybpe $PROCESSED_PATH/$pair.$lg.$split $PARA_PATH/$pair.$lg.$split $CODES_PATH
#            python preprocess.py $VOCAB_PATH $PROCESSED_PATH/$pair.$lg.$split
#        done
#    done
#done
##### end: prepare data to pretrain a language model

### Prepare XNLI data
rm -rf $PROCESSED_PATH/eval/XNLI
mkdir -p $PROCESSED_PATH/eval/XNLI
# apply BPE codes and binarize the XNLI corpora
for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
  for splt in train valid test; do
    if [ "$splt" = "train" ] && [ "$lg" != "en" ]; then
      continue
    fi
    # separate premise, hypothesis and label
    sed '1d' $XNLI_PATH/${lg}.${splt} | cut -f1 > $PROCESSED_PATH/eval/XNLI/f1.tok
    sed '1d' $XNLI_PATH/${lg}.${splt} | cut -f2 > $PROCESSED_PATH/eval/XNLI/f2.tok
    sed '1d' $XNLI_PATH/${lg}.${splt} | cut -f3 > $PROCESSED_PATH/eval/XNLI/${splt}.label.${lg}

    # apply BPE codes
    $FASTBPE applybpe $PROCESSED_PATH/eval/XNLI/${splt}.s1.${lg} $PROCESSED_PATH/eval/XNLI/f1.tok ${CODES_PATH}
    $FASTBPE applybpe $PROCESSED_PATH/eval/XNLI/${splt}.s2.${lg} $PROCESSED_PATH/eval/XNLI/f2.tok ${CODES_PATH}

    # binarize the data with the vocabulary
    # generate files e.g. test.s1.en.[pth]
    python preprocess.py ${VOCAB_PATH} $PROCESSED_PATH/eval/XNLI/${splt}.s1.${lg}
    python preprocess.py ${VOCAB_PATH} $PROCESSED_PATH/eval/XNLI/${splt}.s2.${lg}

    rm $PROCESSED_PATH/eval/XNLI/*.tok*
  done
done