#!/usr/bin/env bash
# raw data is downloaded from yawat: cf. xlm-code/annotated-ted-talks

# $PWD should be XLM-master/, because we need to use several scripts in this directory
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast
CODES_PATH=$PWD/codes_xnli_15
VOCAB_PATH=$PWD/vocab_xnli_15

# --------------------------- tokenize, lowercase, remove accent

INPUT=/Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/annotated-ted-talks/corpus/rosetta-08-June

# todo the tokenization is already done in these annotated corpus

x="enfr-chp1"

# after this step, the token indices don't change
#echo "Lowercase and remove accent"
#for lg in en fr; do
#    cat $INPUT/${x}.${lg} | python3 $LOWER_REMOVE_ACCENT > $INPUT/${x}.${lg}.uncased
#done

# after this step, the token indices will change for certain words
echo "Applying BPE and binarize"
for lg in en fr ; do
    $FASTBPE applybpe $INPUT/${x}.${lg}.bpe $INPUT/${x}.${lg}.uncased ${CODES_PATH}
    python3 preprocess.py ${VOCAB_PATH} $INPUT/${x}.${lg}.bpe
done

##################### for sub10-testData

#for lg in en fr; do
#    cat $INPUT/sub10.${lg} | python3 $LOWER_REMOVE_ACCENT > $INPUT/sub10.${lg}.uncased
#    $FASTBPE applybpe $INPUT/sub10.${lg}.bpe $INPUT/sub10.${lg}.uncased ${CODES_PATH}
#    python3 preprocess.py ${VOCAB_PATH} $INPUT/sub10.${lg}.bpe
#done


##--------------------------- create cross validation data set
# to conduct fine-tuning on this binary dataset

# 1) first, create a dataset csv file:

# be careful for the labels
# Paste command uses the tab delimiter by default for merging the files.
# The delimiter can be changed to any other character by using the -d option.

# ALGO:
# echo -e "English\tFrench\tLabel" > x.csv  (add this line at first)
# paste en fr > 1
# paste 1 label >> x.csv
# rm 1

#echo "English\tFrench\tLabel" > $INPUT/all.csv
#paste $INPUT/all.en.tok $INPUT/all.fr.tok > 1
#paste 1 ${INPUT}/all.label >> $INPUT/all.csv
#rm 1
#
## finally, need to replace " by \" in the text, otherwise error when pandas reads the csv
## sed 's/\"/\\"/g' a.csv > b.csv
#sed 's/\"/\\"/g' ${INPUT}/all.csv > ${INPUT}/all-sed-quote.csv

# 2) use sklearn to generate datasets of cross validation

#if [ ! -d ./hmt/data/cv_data ]; then
#    mkdir ./hmt/data/cv_data
#fi
#
#python3 ./hmt/data/create-CV-dataset.py ${INPUT}/all-sed-quote.csv ./hmt/data/cv_data/tedannote

# --------------------------- apply bpe then binarize

#FILE_PATH=./hmt/data/cv_data/tedannote/
#
#for fold in {1..10}; do
#    for side in source target; do
#        for splt in train valid; do
#            $FASTBPE applybpe $FILE_PATH/${side}_${splt}_${fold}.bpe $FILE_PATH/${side}_${splt}_${fold}.txt ${CODES_PATH}
#            python3 preprocess.py ${VOCAB_PATH} $FILE_PATH/${side}_${splt}_${fold}.bpe
#            rm $FILE_PATH/${side}_${splt}_${fold}.bpe
#        done
#    done
#done

# scp -r /Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master/hmt/data/cv_data/tedannote \
#slurm:~/nmt-corpus/xlm-master/hmt/data/cv_data