#!/usr/bin/env bash

#CORPUS_PATH=$PWD/hmt-corpus
#MACHINE=$CORPUS_PATH/APR
#HUMAN=$CORPUS_PATH/Europarl

##--------------------------- assemble EN and FR parallel sentences from separate files

#if [ ! -d ./hmt/data/orig ]; then
#    mkdir ./hmt/data/orig
#fi

## check how [for x in] lists files in a directory:
## number order in file name: 0, 1, 10, 100, 1000, 1001 ... and EN and FR files have the same prefixes
## so the same order is kept, sentences are parallel
#cpt=0
#for file in $MACHINE/fr/*.txt ; do
#    cpt=$((cpt+1))
#    echo $file
#    echo $cpt
#    if (( $cpt > 10 )); then
#        break
#    fi
#done

#cpt=0
#rm ./hmt/data/orig/machine_en.txt   # 23235 lines
#for file in $MACHINE/en/*.txt ; do
#    cpt=$((cpt+1))
#    cat $file >> ./hmt/data/orig/machine_en.txt
#done

#rm ./hmt/data/orig/machine_fr.txt   # 23235 lines
#for file in $MACHINE/fr/*.txt ; do
#    cat $file >> ./hmt/data/orig/machine_fr.txt
#done
#wc -l ./hmt/data/orig/machine_fr.txt

#rm ./hmt/data/orig/human_en.txt   # 475834 lines (20 times larger than the machine translated corpus)
#for file in $HUMAN/en/*.txt ; do
#    cat $file >> ./hmt/data/orig/human_en.txt
#done
#
#rm ./hmt/data/orig/human_fr.txt   # 475834 lines
#for file in $HUMAN/fr/*.txt ; do
#    cat $file >> ./hmt/data/orig/human_fr.txt
#done

#head -n 23235 ./hmt/data/orig/human_en.txt > ./hmt/data/orig/part_human_en.txt
#head -n 23235 ./hmt/data/orig/human_fr.txt > ./hmt/data/orig/part_human_fr.txt

##--------------------------- split into 70% train and 30% validation datasets (and test on new datasets)

# 23235*70%=16265, 23235*30%=6970

# machine translated
#head -n 16265 ./hmt/data/orig/machine_en.txt > ./hmt/data/orig/machine_en_train.txt
#tail -n 6970 ./hmt/data/orig/machine_en.txt > ./hmt/data/orig/machine_en_valid.txt
#head -n 16265 ./hmt/data/orig/machine_fr.txt > ./hmt/data/orig/machine_fr_train.txt
#tail -n 6970 ./hmt/data/orig/machine_fr.txt > ./hmt/data/orig/machine_fr_valid.txt

# human translated
#head -n 16265 ./hmt/data/orig/part_human_en.txt > ./hmt/data/orig/human_en_train.txt
#tail -n 6970 ./hmt/data/orig/part_human_en.txt > ./hmt/data/orig/human_en_valid.txt
#head -n 16265 ./hmt/data/orig/part_human_fr.txt > ./hmt/data/orig/human_fr_train.txt
#tail -n 6970 ./hmt/data/orig/part_human_fr.txt > ./hmt/data/orig/human_fr_valid.txt

##--------------------------- create cross validation data set

# combine tokenized human|machine en|fr train|valid datasets
# be careful for the labels
# use sklearn to generate datasets of cross validation
#cd ./hmt/data/tokenized
#cat all_en_train.tok all_en_valid.tok > all_en.tok    # in total 46470 lines
#cat all_fr_train.tok all_fr_valid.tok > all_fr.tok
#cd ./hmt/data/binarized/
#cat train.label valid.label > all.label
#mkdir cv_data/
# Paste command uses the tab delimiter by default for merging the files.
# The delimiter can be changed to any other character by using the -d option.
# English	French	Label  (add this line at first, also separated by '\t')
#paste ../tokenized/all_en.tok ../tokenized/all_fr.tok all.label > cv_data/all_en_fr_label.csv
# then use cv_data/create-CV-dataset.py
# need to replace " by \" in the text
# then use pd.read_csv("all_en_fr_label.csv", names=['English', 'French', 'Label'], sep='\t', escapechar='\\', engine='python', header=0)

# apply bpe then binarize
# don't use the variable name "PATH"
#FILE_PATH=/Users/yumingzhai/PycharmProjects/Coling2020/code/XLM-master/hmt/data/binarized/cv_data/EN-FR
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

#---------------------------------------------

#### for debug, create tiny datasets
#for lg in en fr; do
#    for splt in train valid; do
#        $FASTBPE applybpe ./hmt/data/binarized/tiny_${lg}_${splt}.bpe $TOK_PATH/tiny_${lg}_${splt}.tok ${CODES_PATH}
#        python3 preprocess.py ${VOCAB_PATH} ./hmt/data/binarized/tiny_${lg}_${splt}.bpe
#    done
#done

# -------------------------- combine machine and human translation dataset, apply bpe, binarize

#TOK_PATH=$PWD/hmt/data/tokenized
#
#cat $TOK_PATH/machine_en_train.tok $TOK_PATH/human_en_train.tok > $TOK_PATH/all_en_train.tok
#cat $TOK_PATH/machine_fr_train.tok $TOK_PATH/human_fr_train.tok > $TOK_PATH/all_fr_train.tok
#cat $TOK_PATH/machine_en_valid.tok $TOK_PATH/human_en_valid.tok > $TOK_PATH/all_en_valid.tok
#cat $TOK_PATH/machine_fr_valid.tok $TOK_PATH/human_fr_valid.tok > $TOK_PATH/all_fr_valid.tok

#if [ ! -d ./hmt/data/binarized ]; then
#    mkdir ./hmt/data/binarized
#fi
#
## language independent
#for lg in en fr; do
#    for splt in train valid; do
# applybye first argument is its output
#        $FASTBPE applybpe ./hmt/data/binarized/all_${lg}_${splt}.bpe $TOK_PATH/all_${lg}_${splt}.tok ${CODES_PATH}
#        # result: all_${lg}_${splt}.bpe.pth in ./hmt/data/binarized/
#        python3 preprocess.py ${VOCAB_PATH} ./hmt/data/binarized/all_${lg}_${splt}.bpe
#    done
#done
