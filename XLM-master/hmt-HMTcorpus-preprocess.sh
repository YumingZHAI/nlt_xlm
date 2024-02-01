#!/bin/bash
# hmt stands for human-machine-translation (classifier) (26/02/2020)
# (begin the code with preprocessing a small corpus)
# $PWD should be XLM-master/, because we need to use several scripts in this directory

TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

CODES_PATH=$PWD/codes_xnli_15
VOCAB_PATH=$PWD/vocab_xnli_15

############ first tokenize => en|fr.tok, create labels

# --------------------------- tokenize, lowercase, remove accent

if [ ! -d ./hmt/data/orig/tokenized ]; then
   mkdir ./hmt/data/orig/tokenized
fi

INPUT=./hmt/data/orig
OUTPUT=./hmt/data/orig/tokenized

# top30k-europarl top30k-opensubtitles top30k-tedtalks
for corpus in 33669-books; do
   for origin in source fairseq human ; do
       # language dependent
       if [ $origin = "source" ]; then
           lg="en"
       else
           lg="fr"
       fi
       echo "Tokenizing:" $corpus $origin $lg
       cat $INPUT/${corpus}-${origin}.${lg} | $TOKENIZE ${lg} | python3 $LOWER_REMOVE_ACCENT > $OUTPUT/${corpus}-${origin}.${lg}.tok
   done
done

#-------------------------- create label file: machine 0, human 1
# so the dataset:
# en sentence | machine fr sentence | label 0  30k
# en sentence | human fr sentence | label 1    next 30k
# 33669 for books corpus

#europarl opensubtitles tedtalks
for corpus in books; do
   cpt=0
   rm ./hmt/data/orig/${corpus}.label
   touch ./hmt/data/orig/${corpus}.label
   while (($cpt < 33669)) ; do
       echo "machine" >> ./hmt/data/orig/${corpus}.label
       cpt=$((cpt+1))
   done
   while (($cpt < 67338)) ; do
       echo "human" >> ./hmt/data/orig/${corpus}.label
       cpt=$((cpt+1))
   done
done

##--------------------------- create cross validation data set

# 1) first, create a dataset csv file for each corpus:

# be careful for the labels
# Paste command uses the tab delimiter by default for merging the files.
# The delimiter can be changed to any other character by using the -d option.

# ALGO:
# echo -e "English\tFrench\tLabel" > x.csv  (add this line at first)
# paste en machine-fr > 1
# paste en human-fr > 2
# cat 1 2 > 3
# paste 3 label >> x.csv
# rm 1 2 3

if [ ! -d ./hmt/data/orig/pasted ]; then
   mkdir ./hmt/data/orig/pasted
fi

TOK=./hmt/data/orig/tokenized
OUTPUT=./hmt/data/orig/pasted
top30k-europarl top30k-opensubtitles top30k-tedtalks
for corpus in 33669-books; do
   echo "pasting for corpus" ${corpus}
   echo "English\tFrench\tLabel" > $OUTPUT/${corpus}.csv
   paste $TOK/${corpus}-source.en.tok $TOK/${corpus}-fairseq.fr.tok > 1
   paste $TOK/${corpus}-source.en.tok $TOK/${corpus}-human.fr.tok > 2
   cat 1 2 > 3
   paste 3 ./hmt/data/orig/books.label >> $OUTPUT/${corpus}.csv
   rm 1 2 3
done

## finally, need to replace " by \" in the text, otherwise error when pandas reads the csv
## sed 's/\"/\\"/g' a.csv > b.csv
# europarl opensubtitles tedtalks
for corpus in 33669-books; do
   sed 's/\"/\\"/g' ./hmt/data/orig/pasted/${corpus}.csv > ./hmt/data/orig/pasted/${corpus}-sed-quote.csv
done

# 2) use sklearn to generate datasets of cross validation

if [ ! -d ./hmt/data/cv_data ]; then
   mkdir ./hmt/data/cv_data
fi


# opensubtitles europarl tedtalks
for corpus in books ; do
   python3 ./hmt/data/create-CV-dataset.py ./hmt/data/orig/pasted/33669-${corpus}-sed-quote.csv ./hmt/data/cv_data/${corpus}
done

# 3) apply bpe then binarize
# don't use the variable name "PATH", bash command will be influenced

# europarl opensubtitles tedtalks
for corpus in books; do
   FILE_PATH=./hmt/data/cv_data/${corpus}
   for fold in {1..10}; do
       for side in source target; do
           for splt in train valid; do
               $FASTBPE applybpe $FILE_PATH/${side}_${splt}_${fold}.bpe $FILE_PATH/${side}_${splt}_${fold}.txt ${CODES_PATH}
               python3 preprocess.py ${VOCAB_PATH} $FILE_PATH/${side}_${splt}_${fold}.bpe
               rm $FILE_PATH/${side}_${splt}_${fold}.bpe
           done
       done
   done
done







