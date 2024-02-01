#!/usr/bin/env bash

phrase-level dataset: context sentences and phrase pairs data processing steps:

download crp, aln files from yawat

[for 1 file:
    separate-sentences.py: python3 separate-sentences.py subCorpus1.crp sub1.en sub1.fr
    crp2oneLine.py: python3 crp2oneLine.py subCorpus1.crp sub1.txt
    # todo this category-token-indice.py is different than the one for features-based classifier
    category-token-indice.py: python3 category-token-indice.py subCorpus1.aln sub1.txt sub1Types.txt
]

separate source and target sentences in each crp file, and concatenate them to generate """all.en, all.fr"""
    sh batch-concatenate.sh   # (it uses separate-sentences.py)

turn crp files to txt file, in this format: source_sent#target_sent
    # fortunately every sentence has a unique ID
    cat control-corpus.crp subCorpus1.crp subCorpus2.crp subCorpus3.crp subCorpus4.crp subCorpus5.crp \
    subCorpus6.crp subCorpus7.crp subCorpus8.crp subCorpus9.crp subCorpus13.crp subCorpus14.crp subCorpus15.crp > """all.crp"""
    python3 crp2oneLine.py all.crp """all.txt"""

extract aligned phrases/words in each sentence
    cat control-corpus.aln subCorpus1.aln subCorpus2.aln subCorpus3.aln subCorpus4.aln subCorpus5.aln \
    subCorpus6.aln subCorpus7.aln subCorpus8.aln subCorpus9.aln subCorpus13.aln subCorpus14.aln subCorpus15.aln > """all.aln"""
    # if need to remove empty lines in a file: sed -i '.backup' 's/^$/d' file.txt (modify in the file, also create a backup file)
    python3 category-token-indice.py all.aln all.txt """allTypes.txt"""
    # example: 0, 1 # 0 # literal § 3 # 1 # modulation § 4 # 2 # literal §
    # todo I needed to manually remove the lines (here 1702, 1703) which don't have aligned pairs
    # todo !!! because the only pairs inside contain punctuations, and they are filtered out by my script
    # in all.en, all.fr, allTypes.txt. I removed the lines corresponding to the text:
    # 60,000 .
    # dk : 60,000 .

# for coling experiment, 1722 training sentence pairs are kept in total

"""we should be under XLM-master/ to run this script"""
# the token indices don't change between all.en|fr and all.en|fr.uncased (after applying lowercase and removing accent)
# but the indices are changed after applying BPE
# todo for CV: I keep the generated all.en|fr.uncased|bpe. The bpe.pth files will be generated in the end for each fold
# modify this script and run with needed lines
sh preprocess-ted-forTokenIndice.sh

"""adapt token indices between all.${lg}.uncased and all.${lg}.bpe"""
# example:
# EN-bpe: for some time i have been interested in the pl@@ ace@@ bo effect , which might seem like an odd thing for a mag@@ ic@@ ian to be interested in , unless you think of it in the ter@@ ms that i do , which is , `@@ ` something f@@ ake is believed in enough by somebody that it be@@ comes something real . '@@ '
# the goal is to group splitted token indices together
# [0, 1, 2, 3, 4, 5, 6, 7, 8, [9, 10, 11], 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, [23, 24, 25], 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, [38, 39], 40, 41, 42, 43, 44, 45, 46, [47, 48], 49, [50, 51], 52, 53, 54, 55, 56, 57, 58, 59, [60, 61], 62, 63, 64, [65, 66]]
# Final indices after transforming into "</s> en_sent </s>"
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, [10, 11, 12], 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, [24, 25, 26], 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, [39, 40], 41, 42, 43, 44, 45, 46, 47, [48, 49], 50, [51, 52], 53, 54, 55, 56, 57, 58, 59, 60, [61, 62], 63, 64, 65, [66, 67], 68]
# where [10, 11, 12] is one token ("placebo") splitted into three subwords!
"""output: all.${lg}.bpe-indices"""
for lg in en fr ; do
    python3 trace_index_after_bpe.py all.${lg}.bpe all.${lg}.bpe-indices
done

"""generate phrase-alignment data for each sentence"""
#pairs = open("allTypes.txt", "r")
#bpe_en_indices = open("all.en.bpe-indices", "r")
#bpe_fr_indices = open("all.fr.bpe-indices", "r")
"""output = open(\"all.data\", \"w\")"""
# an example in the output all.data: [5, 6]:[1, 2, 3]:literal	[8, 9]:[5]:literal
python3 dataset-positions.py

"""construct balanced data (same number of L and NL in each sentence) and their corresponding binary sentence data"""
# generate balanced.data, balanced.en|fr.bpe
python3 balance_input_data.py

# before, I manually separated corpus and data to train & valid & test: 80%, 10%, 10%.
# But problem: the test acc is much higher than best dev acc
# The bpe.pth files will be generated in the end for each fold
# result: in balanced-ten-fold/
python3 create-phraseLevel-CV-dataset.py

# vary the proportion in training data: need to modify the variables "distribution" and "multiple"
python3 vary_trainData_proportion.py

