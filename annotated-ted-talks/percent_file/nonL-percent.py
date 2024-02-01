# scp vmpakin:/var/www/html/cgi-bin/sample-data/yuming/___ .
# usage: python3 nonL-percent.py ${name}

import sys

name = sys.argv[1]
nl_penc_file = open("./" + name + ".txt", "w")

with open("../corpus/" + name + ".aln") as file:
    # the first line is empty. start from the second line.
    file.readline()
    i = 0
    for l in file:
        # print(l)
        total_src = 0
        nl_src = 0
        tab = l.strip().split()[1:]
        for pair in tab:
            source = pair.split(':')[0]
            label = pair.split(':')[2]
            # don't count unaligned_explicitation
            if source != '':
                l_src = len(source.split(','))
            else:
                l_src = 0
            total_src += l_src
            # unaligned: no type
            # count aligned non-literal cases
            if label != 'literal' and label != 'equivalence' and label != 'lexical_shift' :
                    # and label != 'unaligned' :
                    # and label != 'unaligned_reduction' and label != "uncertain" and label != 'translation_error' :
                nl_src += l_src
                # print(pair)
                # print(l_src)
        # print('total tokens %i' % total_src)
        # print("non-literal translated tokens %i" % nl_src)
        # print('Non-literal translation percent (token level) %.2f%%' % (nl_src*100.0/total_src))
        # print("----------------------------")
        print(nl_src*1.0/total_src, file=nl_penc_file)
        # if the percent is 0, there is no non-literal translations inside
        # i += 1
        # if i > 5:
        #     break
