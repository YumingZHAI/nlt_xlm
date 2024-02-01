import numpy as np
import subprocess

def filter_input_data(file):
    """create balanced dataset; return the IDs of sentences which don't contain any NL translations"""
    # e.g. [2, 3, 4]:[1, 2, 3, 4]:equivalence	[6]:[6]:literal	[7]:[7, 8]:literal	[8]:[9]:literal	[9]:[10]:literal	[10]:[11]:literal	[11, 12]:[15, 16, 17]:literal	[13, 14]:[18, 19]:generalization	[15]:[13, 14]:literal
    # for training:
    # output = open(corpus_path + "balanced.data", "w")
    # for test: todo comment output here for the moment
    # output = open(corpus_path + "sub10.balanced.data", "w")
    del_sentID = []
    sentID = 0
    cumulated_literal = 0
    cumulated_NL = 0
    with open(file, "r") as f:
        for line in f:
            i = 0
            label_list = []
            literal_indices = []
            nonL_indices = []
            line = line.strip()
            tab = line.split("\t")
            for pair in tab:
                label = pair.split(':')[2]
                if label in (["literal", "equivalence", "lexical_shift"]):
                    label_id = 0
                    literal_indices.append(i)
                elif label in (["transposition", "generalization", "particularization", "modulation", "modulation_transposition", "figurative"]):
                    label_id = 1
                    nonL_indices.append(i)
                i += 1
                label_list.append(label_id)
            cumulated_NL += len(nonL_indices)
            # the sentences that don't contain any NL translations
            if not 1 in label_list:
                del_sentID.append(sentID)   # del_sentID contains only these cases
            else:
                # print(tab)
                # print("label list", label_list)
                # print("literal_indices", literal_indices)
                # print("nonL indices", nonL_indices)
                if literal_indices:
                    # if there are more L than NL, randomly take as many L as NL
                    if len(literal_indices) >= len(nonL_indices):
                        filtered_literal = list(np.random.choice(literal_indices, size=len(nonL_indices), replace=False))
                    else:
                        filtered_literal = literal_indices
                    # combine and sort the kept indice list
                    final_list = sorted(nonL_indices + filtered_literal)
                    cumulated_literal += len(filtered_literal)
                # if there are only NL examples
                else:
                    final_list = nonL_indices
                # print(final_list)
                # print("\t".join([tab[x] for x in final_list]), file=output)
            sentID += 1
    # output.close()

    print("cumulated_literal", cumulated_literal)
    print("cumulated_NL", cumulated_NL)
    return del_sentID

# /Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/annotated-ted-talks/corpus/
corpus_path = "./"

# for training:
# del_sentID = filter_input_data(corpus_path + "all.data")
# for test
del_sentID = filter_input_data(corpus_path + "sub10.data")

# todo needed for the COLING oracle study, test dataset
print(del_sentID)
# print(len(del_sentID))
# for lg in ["en", "fr"] :
#     i = 0
#     output = open(corpus_path + "sub10.balanced." + lg, "w")
#     with open(corpus_path + "sub10." + lg, "r") as file:
#         for line in file:
#             if i not in del_sentID:
#                 print(line.strip(), file=output)
#             i += 1
#         output.close()

# todo comment for the moment
# tool_path = "/Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master/"
# VOCAB_PATH = tool_path + "vocab_xnli_15"

# for lg in ["en", "fr"] :
#     i = 0
#     # for training:
#     # output = open(corpus_path + "balanced." + lg + ".bpe", "w")
#     # for test:
#     output = open(corpus_path + "sub10.balanced." + lg + ".bpe", "w")
#     # for training:
#     # with open(corpus_path + "all." + lg + ".bpe", "r") as file:
#     # for test:
#     with open(corpus_path + "sub10." + lg + ".bpe", "r") as file:
#         for line in file:
#             if i not in del_sentID:
#                 print(line.strip(), file=output)
#             i += 1
#         output.close()
#     # only for test:
#     bash_command = "python3 " + tool_path + "preprocess.py "  + VOCAB_PATH + ' ' + corpus_path + "sub10.balanced." + lg + ".bpe"
#     process = subprocess.Popen(bash_command.split(), stdout = subprocess.PIPE)
#     output, error = process.communicate()