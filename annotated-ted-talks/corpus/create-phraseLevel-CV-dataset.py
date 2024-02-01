import pandas as pd
import sys
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import subprocess

# generate line indices of a dataset separated into ten folds

# balanced dataset contains 1110 lines in total
# output = open("balanced-lines.csv", "w")
# for i in range(0, 1110):
#     if i == 0:
#         print("header", file=output)
#     print(i, "\t label", file=output)
# output.close()
#
# dataframe = pd.read_csv("balanced-lines.csv", sep='\t', engine='python', header=0)
# X = dataframe.values[:, 0:-1]
# y = dataframe.values[:, -1]
#
# skf = StratifiedKFold(n_splits=10, shuffle=True)
#
# name_dir = "balanced-ten-fold"
# if not os.path.exists(name_dir):
#     os.makedirs(name_dir)
#
# with open("balanced.en.bpe", "r") as f:
#     en_sent_lines = f.readlines()
#
# with open("balanced.fr.bpe", "r") as f:
#     fr_sent_lines = f.readlines()
#
# with open("balanced.data", "r") as f:
#     data_lines = f.readlines()
#
# j = 1
# for train_index, valid_index in skf.split(X,y):
#     # print(train_index, valid_index)
#
#     en_sent_train = open(name_dir + "/balanced.train" + str(j) + ".en.bpe", "w")
#     fr_sent_train = open(name_dir + "/balanced.train" + str(j) + ".fr.bpe", "w")
#     phraseData_train = open(name_dir + "/balanced.train" + str(j) + ".data", "w")
#     for index in train_index:
#         print(en_sent_lines[index].strip(), file=en_sent_train)
#         print(fr_sent_lines[index].strip(), file=fr_sent_train)
#         print(data_lines[index].strip(), file=phraseData_train)
#
#     en_sent_valid = open(name_dir + "/balanced.valid" + str(j) + ".en.bpe", "w")
#     fr_sent_valid = open(name_dir + "/balanced.valid" + str(j) + ".fr.bpe", "w")
#     phraseData_valid = open(name_dir + "/balanced.valid" + str(j) + ".data", "w")
#     for index in valid_index:
#         print(en_sent_lines[index].strip(), file=en_sent_valid)
#         print(fr_sent_lines[index].strip(), file=fr_sent_valid)
#         print(data_lines[index].strip(), file=phraseData_valid)
#
#     j += 1
# in each fold, the result files contain [approximately] equivalent L and NL instances

tool_path = "/Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master/"
VOCAB_PATH = tool_path + "vocab_xnli_15"

# for lg in ["en", "fr"]:
#     for splt in ["train", "valid"]:
#         for i in range(1, 11):
#             bash_command = "python3 " + tool_path + "preprocess.py "  + VOCAB_PATH + ' ' + name_dir + "/balanced." + splt + str(i) + "." + lg + ".bpe"
#             process = subprocess.Popen(bash_command.split(), stdout = subprocess.PIPE)
#             output, error = process.communicate()

## had error when processing the fold10 sentence data
name_dir = "balanced-ten-fold"

for lg in ["en", "fr"]:
    for splt in ["train", "valid"]:
        i = 10
        bash_command = "python3 " + tool_path + "preprocess.py "  + VOCAB_PATH + ' ' + name_dir + "/balanced." + splt + str(i) + "." + lg + ".bpe"
        process = subprocess.Popen(bash_command.split(), stdout = subprocess.PIPE)
        output, error = process.communicate()
