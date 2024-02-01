import os, sys

# python3 category-token-indice.py x.aln x.txt (obtained from crp2oneLine.py) xTypes.txt

# (30/03/2020 for coling submission) I modify this script to get for each sentence pair:
# a group of aligned phrases and their label
# don't include uncertain, translation_error, reduction, explicitation, punctuation

ALIGNMENT = sys.argv[1]
CORPUS = sys.argv[2]  # en_sent#fr_sent
OUTPUT = open(sys.argv[3], "w")   # x_types.txt

# if you have changed some tokenization in file.crp, you must also change the file.txt
# attention: after saving modifications on yawat, the first line of file.aln is always empty

# simultaneously loop over aln and txt file
# for each sentence, generate a list of all pairs
for alignment, pair in zip(open(os.path.abspath(ALIGNMENT)), open(os.path.abspath(CORPUS))):
    allPairs = []
    if not alignment == "\n":
        # ['id' '0:0:literal' '1:1:literal' ...]
        tabAlign = alignment.split()
        sentenceID = tabAlign[0]
        # two sentences in one line, separated by '#'
        sentences = pair.split('#')
        sourceWords = sentences[0].split()
        targetWords = sentences[1].split()
    else:
        tabAlign = []
    for align in tabAlign:  # 0,1,3,2:4,0,1,2,3:transposition
        sourceSegment = ""
        targetSegment = ""
        label = ""

        if ':' in align:  # don't take the first id number of sentences
            # e.g. 0:0:literal or 8,9,10:8:modulation. There exist unaligned source or target words
            components = align.split(':')
            label = components[2]

            # n words in source
            if ',' in components[0]:
                continuNumsSource = components[0].split(',')
                # [8, 9, 10] there exists also discontinuous cases
                # there exists also the cases of: [10, 8, 9], so we need sort()
                # convert list of string to int
                continuNumsSource = map(int, continuNumsSource)
                # sort word index to get the right order
                continuNumsSource = sorted(continuNumsSource)

                for num in continuNumsSource:
                    sourceSegment += sourceWords[int(num)] + " "

                # to get token & indice for words in a segment
                sourceSegment = sourceSegment.strip(" $") + "_" + str(continuNumsSource).strip('[]')

                # 1 word in target
                if ',' not in components[1]:
                    if components[1] != "":
                        targetSegment = targetWords[int(components[1])] + "_" + str(components[1]).strip('[]')
                    else:
                        targetSegment = "null"
                # n word in target
                else:
                    continuNumsTarget = components[1].split(',')
                    continuNumsTarget = map(int, continuNumsTarget)
                    continuNumsTarget = sorted(continuNumsTarget)
                    for num in continuNumsTarget:
                        targetSegment += targetWords[int(num)] + " "
                    targetSegment = targetSegment.strip(" $") + "_" + str(continuNumsTarget).strip('[]')

            # 1 word in source
            if ',' not in components[0]:
                if components[0] != "":
                    sourceSegment = sourceWords[int(components[0])] + "_" + str(components[0]).strip('[]')
                else:
                    sourceSegment = "null"
                # n words in target
                if ',' in components[1]:
                    continuNumsTarget = components[1].split(',')
                    continuNumsTarget = map(int, continuNumsTarget)
                    continuNumsTarget = sorted(continuNumsTarget)
                    for num in continuNumsTarget:
                        targetSegment += targetWords[int(num)] + " "
                    targetSegment = targetSegment.strip(" $") + "_" + str(continuNumsTarget).strip('[]')
                # 1 word in target
                else:
                    if components[1] != "":
                        targetSegment = targetWords[int(components[1])] + "_" + str(components[1]).strip('[]')
                    else:
                        targetSegment = "null"

        if sourceSegment != "" and targetSegment != "":
            if not any(punct in sourceSegment.split('_')[0] for punct in ".,!?"):
                if len(sourceSegment.split('_')) == 2 and len(targetSegment.split('_')) == 2:
                    # todo only keep token indices
                    pairSegment = sourceSegment.split('_')[1] + " # " + targetSegment.split('_')[1]

                    if label == "literal":
                        allPairs.append(pairSegment + " # " + "literal")
                    elif label == "localization":
                        allPairs.append(pairSegment + " # " + "localization")
                    elif label == "modulation":
                        allPairs.append(pairSegment + " # " + "modulation")

    if not alignment == "\n":
        for value in allPairs:
            print(value, file=OUTPUT, end=" § ")
        print(file=OUTPUT)

OUTPUT.close()