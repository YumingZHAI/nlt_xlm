import ast

# for test
pairs = open("enfr-chp1-Types.txt", "r")
bpe_en_indices = open("enfr-chp1.en.bpe-indices", "r")
bpe_fr_indices = open("enfr-chp1.fr.bpe-indices", "r")
output = open("enfr-chp1.data", "w")

# for training
# pairs = open("allTypes.txt", "r")
# bpe_en_indices = open("all.en.bpe-indices", "r")
# bpe_fr_indices = open("all.fr.bpe-indices", "r")
# output = open("all.data", "w")

def flattened_indices(orig_ind, bpe_indice):
    a = []
    for i in orig_ind.split(','):
        new_i = bpe_indice[int(i) + 1]
        a.append(new_i)
    flattened = []
    flag = False
    for elem in a:
        if isinstance(elem, list):
            flag = True
    if flag == True:
        for elem in a:
            if isinstance(elem, int):
                flattened.append(elem)
            else:
                for each in elem:
                    flattened.append(each)
    elif flag == False:
        flattened = a
    return flattened

for pair, bpe_en_indice, bpe_fr_indice in zip(pairs, bpe_en_indices, bpe_fr_indices):
    tabs = pair.strip().strip('§\n').split(' § ')
    bpe_en_indice = ast.literal_eval(bpe_en_indice.strip())
    bpe_fr_indice = ast.literal_eval(bpe_fr_indice.strip())
    for tab in tabs:
        src_ind, tgt_ind, label = tab.split(' # ')
        flattened_src = flattened_indices(src_ind, bpe_en_indice)
        flattened_tgt = flattened_indices(tgt_ind, bpe_fr_indice)
        print(str(flattened_src) + ":" + str(flattened_tgt) + ":" + label, file=output, end="\t")
    print(file=output)

output.close()