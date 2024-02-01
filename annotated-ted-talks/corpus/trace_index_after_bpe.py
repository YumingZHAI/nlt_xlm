# 27/03/2020 take a long time to finish, worth sharing with others
# python3 trace_index_after_bpe.py all.${lg}.bpe all.${lg}.bpe-indices(output) (language independent)
import sys
output = open(sys.argv[2], "w")

with open(sys.argv[1], "r") as file:
    for l in file:
        l = l.strip()
        # for some time i have been interested in the pl@@ ace@@ bo effect ,
        # print(l)
        i = 0
        raw_list = []
        for subword in l.split():
            if '@@' not in subword:
                raw_list.append(i)
            else:
                split_list = []
                split_list.append(i)
                # each subword containing '@@', the last subword without '@@' is ignored for the moment
                raw_list.append(split_list)
            i+=1
        # print("Raw list, where each index of splitted subword containing '@@' is in a list:")
        # print(raw_list)
        # print("----------------")

        bpe_indices = []
        all_tmp_list = []
        for cpt, element in enumerate(raw_list):
            if isinstance(element, int):
                y = set()
                for l in all_tmp_list:
                    for x in l:
                        y.add(x)
                if element not in y:   # append the indices of words which are not part of splitted words
                    bpe_indices.append(element)
            elif isinstance(element, list):
                tmp_list = []
                k = 0
                while isinstance(raw_list[cpt+k], list):   # if the next element(s) is/are also lists
                    tmp_list.extend(raw_list[cpt+k])
                    k+=1
                tmp_list.append(raw_list[cpt+k])   # the last subword (without '@@'), which is not in a list
                result = False
                for previous_list in all_tmp_list:
                    result = all(elem in previous_list for elem in tmp_list)
                if result == False:
                    bpe_indices.append(tmp_list)    # don't include sublists, e.g. [10,11] is a sublist of [9,10,11]
                all_tmp_list.append(tmp_list)    # first, put into [9,10,11], next time, [10,11] will be its sublist
        # print("New list: group splitted token indices together")
        # print(bpe_indices)
        # print("----------------")

        # transform after transforming into "</s> sent </s>"
        new = []
        for elem in bpe_indices:
            if isinstance(elem, int):
                new.append(elem + 1)
            elif isinstance(elem, list):
                new.append([x + 1 for x in elem])
        last_element = new[-1]
        if isinstance(last_element, int):
            final = [0] + new + [last_element + 1]
        elif isinstance(last_element, list):
            final = [0] + new + [last_element[-1] + 1]
        # print("Final indices after adding </s> </s>")
        print(final, file=output)

output.close()