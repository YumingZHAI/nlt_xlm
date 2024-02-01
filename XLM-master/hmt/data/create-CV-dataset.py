import pandas as pd
import sys
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# split a data set ('\t' separated) to ten folds (train+valid)
# input: ${corpus}-sed-quote.csv

corpus = sys.argv[1]

dataframe = pd.read_csv(corpus, names=['English', 'French', 'Label'], sep='\t', escapechar='\\', engine='python', header=0)
array = dataframe.values
X = array[:, 0:-1]   # English and French sentences
y = array[:, -1]     # labels     # if label are numbers, put .astype('int'), otherwise error in spliting
# count the number of instances for each class
# print(Counter(y))

skf = StratifiedKFold(n_splits=10)

name_dir = sys.argv[2]
if not os.path.exists(name_dir):
    os.makedirs(name_dir)

j = 1
for train_index, valid_index in skf.split(X, y):
    src_train = open(name_dir + "/source_train_" + str(j) + ".txt", "w")
    tgt_train = open(name_dir + "/target_train_" + str(j) + ".txt", "w")
    label_train = open(name_dir + "/label_train_" + str(j) + ".txt", "w")
    for index in train_index:
        print(X[index][0], file=src_train)
        print(X[index][1], file=tgt_train)
        print(y[index], file=label_train)

    src_valid = open(name_dir + "/source_valid_" + str(j) + ".txt", "w")
    tgt_valid = open(name_dir + "/target_valid_" + str(j) + ".txt", "w")
    label_valid = open(name_dir + "/label_valid_" + str(j) + ".txt", "w")
    for index in valid_index:
        print(X[index][0], file=src_valid)
        print(X[index][1], file=tgt_valid)
        print(y[index], file=label_valid)

    j += 1