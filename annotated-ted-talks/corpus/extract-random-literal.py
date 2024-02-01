import ast

with open("sub10.balanced.data") as data, open("sub10.balanced.en.bpe") as en, open("sub10.balanced.fr.bpe") as fr:
    for d, e, f in zip(data, en, fr):
        tab = d.strip().split('\t')
        for x in tab:
            source, target, label = x.split(':')
            if label == 'literal' or label == 'equivalence' or label == 'lexical_shift':
                # print(x)
                src_tab = ast.literal_eval(source)
                tgt_tab = ast.literal_eval(target)
                for i in src_tab:
                    print(e.split()[i-1], end=" ")
                print("->", end=" ")
                for i in tgt_tab:
                    print(f.split()[i-1], end=" ")
                print("(" + label +  ")")
        print("---------")
