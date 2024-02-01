import numpy as np
import subprocess

def filter_input_data(input_data, output, multiple):
    """create training dataset with different data distribution, e.g. 5NL vs 1NL (in terms of nb of phrase pairs)"""
    sentID = 0
    cumulated_literal = 0
    cumulated_NL = 0
    literal_dico = {}
    nonL_sentID = []
    with open(input_data, "r") as f:
        # input data format: [2, 3, 4]:[1, 2, 3, 4]:equivalence	[6]:[6]:literal	[7]:[7, 8]:literal
        for line in f:
            line = line.strip()
            tab = line.split("\t")
            i = 0
            # gold labels of each phrase pair (0 or 1)
            label_list = []
            # phrase pair indice in a sentence (0 - N)
            literal_indices = []
            nonL_indices = []
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
            # for the sentences that don't contain any NL translations, construct a dictionary:
            # (key) sentenceID : (value) nb of phrase pairs contained inside
            if not 1 in label_list:
                literal_dico[sentID] = len(label_list)
            # if the sentence contains NL pairs
            else:
                # print(tab)
                # print("label list", label_list)
                # print("literal_indices", literal_indices)
                # print("nonL indices", nonL_indices)
                nonL_sentID.append(sentID)
                # if there are literal AND non-literal pairs
                if literal_indices:
                    if len(literal_indices) >= multiple*len(nonL_indices):
                        filtered_literal = list(np.random.choice(literal_indices, size=multiple*len(nonL_indices), replace=False))
                    # e.g. if L's number < 5*NL's number, take all of L
                    else:
                        filtered_literal = literal_indices
                    # combine and sort
                    final_list = sorted(nonL_indices + filtered_literal)
                    cumulated_literal += len(filtered_literal)
                # if there are only non-literal pairs
                else:
                    final_list = nonL_indices
                print("\t".join([tab[x] for x in final_list]), file=output)
            sentID += 1

    # print(cumulated_literal)
    # print(cumulated_NL)
    # after the above filtering, we still need these nbs of Literal pairs to get e.g. 5L:1NL
    still_need = multiple*cumulated_NL - cumulated_literal
    sorted_tuple = sorted(literal_dico.items(), key=lambda x: x[1], reverse=True)

    pureLiteral_sentID_list = []
    pure_literal = 0
    for x in sorted_tuple:
        if pure_literal >= still_need:
            break
        pure_literal += x[1]
        pureLiteral_sentID_list.append(x[0])

    # print(still_need)
    # print(pure_literal)
    # print(pureLiteral_sentID_list)
    # print(len(pureLiteral_sentID_list))
    ## don't sort, keep this order!
    kept_sentID_list = nonL_sentID + pureLiteral_sentID_list
    # print(len(kept_sentID_list))
    return pureLiteral_sentID_list, kept_sentID_list

##################### generate the first part of the .data file
splt = "train"
distribution = "2_1"
multiple = 2
corpus_path = "./"   # /Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/annotated-ted-talks/corpus/
inputfile_path = corpus_path + "all." + splt + ".data"
outputfile_path = corpus_path + distribution + "." + splt + ".data"

output = open(outputfile_path, "w")
pureLiteral_sentID_list, kept_sentID_list = filter_input_data(inputfile_path, output, multiple)

output.close()

##################### complete the .data file:
with open(inputfile_path, "r") as f:
    a = f.readlines()
output = open(outputfile_path, "a")  # append mode
for x in pureLiteral_sentID_list:
    print(a[x].strip(), file=output)
output.close()

##################### generate the .bpe.pth file
tool_path = "/Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master/"
VOCAB_PATH = tool_path + "vocab_xnli_15"

for lg in ["en", "fr"] :
    with open(corpus_path + "all." + splt + "." + lg + ".bpe", "r") as f:
        a = f.readlines()

    file_path = corpus_path + distribution + "." + splt + "." + lg + ".bpe"   # for output
    output = open(file_path, "w")
    for x in kept_sentID_list:
        print(a[x].strip(), file=output)
    output.close()

    # generate .bpe.pth file
    bash_command = "python3 " + tool_path + "preprocess.py "  + VOCAB_PATH + ' ' + file_path
    process = subprocess.Popen(bash_command.split(), stdout = subprocess.PIPE)
    output, error = process.communicate()