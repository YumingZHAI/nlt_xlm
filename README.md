# nlt_xlm
This repository contains the code and dataset used in this article: 

Detecting Non-literal Translations by Fine-tuning Cross-lingual Pre-trained Language Models, which is published in COLING2020. 

XLM original code: https://github.com/facebookresearch/XLM

Scripts adapted by me in this project: 

1)Classify human-vs-machine translation

`xlm-code/XLM-master/hmt-HMTcorpus-preprocess.sh`   

`xlm-code/XLM-master/run_ft_hmt.sh`

`xlm-code/XLM-master/finetune-hmt.py` 

`xlm-code/XLM-master/src/evaluation/hmt.py` 

2)Detect phrase-level non-literal translation

`xlm-code/annotated-ted-talks/corpus/phrase-level-pipeline.sh`

`xlm-code/annotated-ted-talks/corpus/trace_index_after_bpe.py` (adjust source and target phrase alignment token indices to BPE-ized token indices)

`xlm-code/XLM-master/src/evaluation/nltClassifer.py` (according to these subword indices, get source and target segment XLM representations by a max-pooling of their embeddings, do feature engineering as in Arase et al. and modify the forward function)

`xlm-code/XLM-master/src/evaluation/phraseTT.py` (construct dataset + labels by batch, to run the fine-tuning: before, each sentence pair corresponds to one label; now, each pair contains many phrase pairs, each corresponding to one label)

`xlm-code/annotated-ted-talks/corpus/balance_input_data.py` (resolve the imbalanced dataset problem: create balanced dev+test set)

`xlm-code/annotated-ted-talks/corpus/vary_trainData_proportion.py` (Vary the Literal versus Non-Literal proportion in training data) 

