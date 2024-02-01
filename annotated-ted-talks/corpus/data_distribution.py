from collections import Counter

with open("all.test.data", "r") as file:
    labels = []
    for line in file:
        tabs = line.strip().split("\t")
        for tab in tabs:
            label = tab.split(':')[2]
            if label in (["literal", "equivalence", "lexical_shift"]):
                label_id = 0
            elif label in (["transposition", "generalization", "particularization", "modulation", "modulation_transposition", "figurative"]):
                label_id = 1
            labels.append(label_id)

    print(len(labels))
    print(Counter(labels))
    # train data:
    # 20108
    # Counter({0: 18403, 1: 1705})  92% literal
    # dev data:
    # 1617
    # Counter({0: 1443, 1: 174})   89% literal
    # test data:
    # 2300
    # Counter({0: 1945, 1: 355})   85% literal
