#!/usr/bin/env bash

# for saving
scp vmpakin:/var/www/html/cgi-bin/sample-data/huck/enfr-chp1.* /Applications/MAMP/htdocs/yawat/sample-data/demo
# for processing
scp vmpakin:/var/www/html/cgi-bin/sample-data/huck/enfr-chp1.* /Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/annotated-ted-talks/corpus/rosetta-08-June

python3 ../separate-sentences.py enfr-chp1.crp enfr-chp1.en enfr-chp1.fr

python3 ../crp2oneLine.py enfr-chp1.crp enfr-chp1.txt

# todo this category-token-indice.py is different than the one for features-based classifier
# todo must pay attention here
python3 ../rosetta-category-token-indice.py enfr-chp1.aln enfr-chp1.txt enfr-chp1-Types.txt

#under XLM-master/
sh preprocess-ted-forTokenIndice.sh
# output: *.en|fr.uncased|bpe(.pth)

# adapt token indices between *.${lg}.uncased and *.${lg}.bpe
# output: *.${lg}.bpe-indices
for lg in en fr ; do
    python3 ../trace_index_after_bpe.py enfr-chp1.${lg}.bpe enfr-chp1.${lg}.bpe-indices
done

#generate phrase-alignment data for each sentence
#need to put the correct filenames inside
python3 ../dataset-positions.py
# output: enfr-chp1.data

# todo .data: problem of too long paragraphs
scp enfr-chp1.data enfr-chp1.en.bpe.pth enfr-chp1.fr.bpe.pth \
slurm:~/nmt-corpus/annotated-ted-talks/corpus/rosetta-testData

scp /Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master/src/evaluation/phraseTT.py \
slurm:~/nmt-corpus/xlm-master/src/evaluation/

scp /Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master/run_ft_nlTrans.sh \
slurm:~/nmt-corpus/xlm-master/

81900 with the entire file enfr-chp1.data
#=> IndexError: index 560 is out of bounds for dimension 0 with size 512

#remove for the moment the phrase pairs which have indexes larger than 256
scp enfr-chp1.data slurm:~/nmt-corpus/annotated-ted-talks/corpus/rosetta-testData

# todo the label are changed! literal vs [localization & modulation]
81903 correct bug in rosetta-category-token-indice.py
81908 update phraseTT.py, in test function


result: 19/29 65.5%
prediction, then gold label

tensor([0, 1], device='cuda:0')
tensor([1, 1], device='cuda:0')
correctly predicted 1 / total 2

tensor([1, 1, 1], device='cuda:0')
tensor([1, 1, 1], device='cuda:0')
correctly predicted 3 / total 3

tensor([1, 0, 0, 1, 0], device='cuda:0')
tensor([1, 1, 1, 1, 1], device='cuda:0')
correctly predicted 2 / total 5

tensor([0, 1], device='cuda:0')
tensor([1, 1], device='cuda:0')
correctly predicted 1 / total 2

tensor([0, 0, 1, 1], device='cuda:0')
tensor([1, 1, 1, 1], device='cuda:0')
correctly predicted 2 / total 4

tensor([1, 1, 0], device='cuda:0')
tensor([1, 1, 1], device='cuda:0')
correctly predicted 2 / total 3

tensor([1, 0, 1], device='cuda:0')
tensor([1, 1, 1], device='cuda:0')
correctly predicted 2 / total 3

tensor([1, 0, 1, 1], device='cuda:0')
tensor([1, 1, 1, 1], device='cuda:0')
correctly predicted 3 / total 4

tensor([1, 1, 1], device='cuda:0')
tensor([1, 1, 1], device='cuda:0')
correctly predicted 3 / total 3



