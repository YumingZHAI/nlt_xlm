#!/usr/bin/env bash

nonL-percent.py:
input: ../corpus/x.aln
output: x.txt -> percent of non-literal translations in a sentence

concatenate-percent.sh:
use "nonL-percent.py" to process all .aln files
output: all_percent.txt    # 545 0.0 percent

create-label.py:
input: all_percent.txt
output: all.label -> absent | present

correlation.py:
input: all_percent.txt, [corpusName]_sentenceLevel_clf_result.txt  # hmt load test/all.en.bpe.pth and write prediction
output: correlation value, plot