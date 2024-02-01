#!/usr/bin/env bash
python3 separate-sentences.py control-corpus.crp en-control fr-control
python3 separate-sentences.py subCorpus1.crp en1 fr1
python3 separate-sentences.py subCorpus2.crp en2 fr2
python3 separate-sentences.py subCorpus3.crp en3 fr3
python3 separate-sentences.py subCorpus4.crp en4 fr4
python3 separate-sentences.py subCorpus5.crp en5 fr5
python3 separate-sentences.py subCorpus6.crp en6 fr6
python3 separate-sentences.py subCorpus7.crp en7 fr7
python3 separate-sentences.py subCorpus8.crp en8 fr8
python3 separate-sentences.py subCorpus9.crp en9 fr9
python3 separate-sentences.py subCorpus13.crp en13 fr13
python3 separate-sentences.py subCorpus14.crp en14 fr14
python3 separate-sentences.py subCorpus15.crp en15 fr15

# the same order as in percent_file/concatenate-percent.py
cat en-control en1 en2 en3 en4 en5 en6 en7 en8 en9 en13 en14 en15 > all.en
cat fr-control fr1 fr2 fr3 fr4 fr5 fr6 fr7 fr8 fr9 fr13 fr14 fr15 > all.fr

rm [enfr]*-control
rm en[0-9]*
rm fr[0-9]*
