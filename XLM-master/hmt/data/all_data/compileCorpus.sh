#cd /Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master/hmt/data/cv_data/europarl
#cat label_train_1.txt label_valid_1.txt > ../../all_data/E or S or T/label_all.txt
#cat source_train_1.txt source_valid_1.txt > PATH/source_all.txt
#cat target_train_1.txt target_valid_1.txt > PATH/target_all.txt

# cd xlm-code/XLM-master/hmt/data/all_data
WORK_PATH="/Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master"
TOOLS_PATH=$WORK_PATH/tools

FASTBPE=$TOOLS_PATH/fastBPE/fast
CODES_PATH=$WORK_PATH/codes_xnli_15
VOCAB_PATH=$WORK_PATH/vocab_xnli_15

# europarl opensubtitles tedtalks
for corpus in books ; do
    FILE_PATH=${corpus}
    for side in source target; do
        $FASTBPE applybpe $FILE_PATH/${side}_all.bpe $FILE_PATH/${side}_all.txt ${CODES_PATH}
        python3 $WORK_PATH/preprocess.py ${VOCAB_PATH} $FILE_PATH/${side}_all.bpe
    done
done

