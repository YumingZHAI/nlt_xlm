python3 finetune-hmt.py  \
--exp_name fineTune_xlm_hmt  \
--dump_path ./dumped         \
--model_path ./models/mlm_tlm_xnli15_1024.pth  \
--data_path ./hmt/data    \
--transfer_tasks HMT          \
--trainCorpus books  \
--devCorpus europarl \
--inference False       \
--batch_size 4            \
--resume_training False  \
--clf_result sentenceLevel_clf_result.txt \
--checkpoint_dir ./sent-checkpoint/  \
--optimizer_e adam,lr=0.000005        \
--optimizer_p adam,lr=0.000005        \
--finetune_layers "0:_1"              \
--n_epochs 10                         \
--epoch_size 24000                   \
--max_len 256                        \
--max_vocab 95000    \
--fold $1                  \

# inference False, batch_size 4; otherwise, 1

# 2.28日 跑实验时看到epoch_size 确实可以不用设为-1, 因为loss已经没有再降低的趋势

#--------------------------------------------- learning process
# "We provide large pretrained models for the 15 languages of XNLI, and two other models in 17 and 100 languages."
# I use the pre-trained model mlm_tlm_xnli15_1024.pth

# XNLI 15 languages: English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili and Urdu
# XNLI dataset: No specific tokenizers for language ar, bg, sw, tr, ur, vi
# (ar:Arabic, bg:Bulgarian, el: Greek, hi: Hindi, sw: Swahili, tr: Turkish, ur: Urdu, vi: Vietnamese)

#python glue-xnli.py
#--exp_name test_xnli_mlm_tlm             # experiment name
#--dump_path ./dumped/                    # where to store the experiment
#--model_path mlm_tlm_xnli15_1024.pth     # model location
#--data_path ./data/processed/XLM15       # data location
#--transfer_tasks XNLI,SST-2              # transfer tasks (XNLI or GLUE tasks)
#--optimizer_e adam,lr=0.000025           # optimizer of projection (lr \in [0.000005, 0.000025, 0.000125])
#--optimizer_p adam,lr=0.000025           # optimizer of projection (lr \in [0.000005, 0.000025, 0.000125])
#--finetune_layers "0:_1"                 # fine-tune all layers
#--batch_size 8                           # batch size (\in [4, 8])
#--n_epochs 250                           # number of epochs
#--epoch_size 20000                       # number of sentences per epoch
#--max_len 256                            # max number of words in sentences
#--max_vocab 95000                        # max number of words in vocab

#indeed you don't need 250 epochs. We're used to watching the results of our jobs regularly so in general we simply specify a
#high number of epochs and kill jobs manually when it has converged. But you can definitely decrease the number of epochs if you'd like.

# I used this command to understand their program
#python3 glue-xnli.py  \
#--exp_name test_xnli_mlm_tlm  \
#--dump_path ./dumped/         \
#--model_path ./models/mlm_tlm_xnli15_1024.pth  \
#--data_path ./data/processed/XLM15    \
#--transfer_tasks XNLI                 \
#--optimizer_e adam,lr=0.000025        \
#--optimizer_p adam,lr=0.000025        \
#--finetune_layers "0:_1"              \
#--batch_size 8                        \
#--n_epochs 250                        \
#--epoch_size 20000                    \
#--max_len 256                         \
#--max_vocab 95000


