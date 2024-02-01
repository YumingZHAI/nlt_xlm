#!/usr/bin/env bash
# raw data is downloaded from yawat (annotated tedtalks)

# $PWD should be XLM-master/, because we need to use several scripts in this directory
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast
CODES_PATH=$PWD/codes_xnli_15
VOCAB_PATH=$PWD/vocab_xnli_15

# copy all.en, all.fr from annotated-ted-talks/corpus into hmt/data/test  (complete 1724 lines)
# copy all.label      from annotated-ted-talks/percent_file into hmt/data/test
# labels: absent 0, present 1

# --------------------------- tokenize, lowercase, remove accent

INPUT=./hmt/data/test

# tokenizing is already done in these annotated corpus
echo "Lowercasing and remove accent"
for lg in en fr; do
   cat $INPUT/all.${lg} | python3 $LOWER_REMOVE_ACCENT > $INPUT/all.${lg}.tok
done

##--------------------------- create cross validation data set
# to conduct fine-tuning on this binary dataset

# 1) first, create a dataset csv file:

# be careful for the labels
# Paste command uses the tab delimiter by default for merging the files.
# The delimiter can be changed to any other character by using the -d option.

#### ALGO:
# echo -e "English\tFrench\tLabel" > x.csv  (add this line at first)
# paste en fr > 1
# paste 1 label >> x.csv
# rm 1

echo "English\tFrench\tLabel" > $INPUT/all.csv
paste $INPUT/all.en.tok $INPUT/all.fr.tok > 1
paste 1 ${INPUT}/all.label >> $INPUT/all.csv
rm 1

### finally, need to replace " by \" in the text, otherwise error when pandas reads the csv
### sed 's/\"/\\"/g' a.csv > b.csv
#sed 's/\"/\\"/g' ${INPUT}/all.csv > ${INPUT}/all-sed-quote.csv

# 2) use sklearn to generate datasets of cross validation

if [ ! -d ./hmt/data/cv_data ]; then
   mkdir ./hmt/data/cv_data
fi

python3 ./hmt/data/create-CV-dataset.py ${INPUT}/all-sed-quote.csv ./hmt/data/cv_data/tedannote

# --------------------------- apply bpe then binarize

echo "Applying BPE and binarize"

for lg in en fr ; do
    $FASTBPE applybpe $INPUT/all.${lg}.bpe $INPUT/all.${lg}.tok ${CODES_PATH}
    python3 preprocess.py ${VOCAB_PATH} $INPUT/all.${lg}.bpe
done

FILE_PATH=./hmt/data/cv_data/tedannote/

for fold in {1..10}; do
    for side in source target; do
        for splt in train valid; do
            $FASTBPE applybpe $FILE_PATH/${side}_${splt}_${fold}.bpe $FILE_PATH/${side}_${splt}_${fold}.txt ${CODES_PATH}
            python3 preprocess.py ${VOCAB_PATH} $FILE_PATH/${side}_${splt}_${fold}.bpe
            rm $FILE_PATH/${side}_${splt}_${fold}.bpe
        done
    done
done

# scp -r /Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master/hmt/data/cv_data/tedannote \
#slurm:~/nmt-corpus/xlm-master/hmt/data/cv_data