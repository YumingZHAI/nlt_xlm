import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# $PWD=/percent_file
# scp slurm:~/nmt-corpus/xlm-master/sentenceLevel_clf_result.txt opensubtitles_sentenceLevel_clf_result.txt

x = []
y = []
with open("all_percent.txt", "r") as file:
    for l in file:
        x.append(float (l.strip()))
with open("books_MT_prediction_proba", "r") as file:
    for l in file:
        y.append(float (l.strip()))

# print(x)
# print(y)

# pearson
print(np.corrcoef(x, y))
# spearman:
print(spearmanr(x, y))

# matplotlib.style.use('ggplot')
plt.scatter(x, y)
plt.show()