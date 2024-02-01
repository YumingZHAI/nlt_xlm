# transform one .crp file to two files containing source and target sentences
# be sure that the ordrer is the same with the corresponding .aln file

# e.g. python3 separate-sentences.py subCorpus1.crp sub1.en sub1.fr
import os, sys, re

input = sys.argv[1]
source = open(sys.argv[2], "w")
target = open(sys.argv[3], "w")

with open(os.path.abspath(input)) as corpus:
    for line in corpus:
        if re.match(r'\d+\n', line):
            print(next(corpus).strip(), file=source)
            print(next(corpus).strip(), file=target)

source.close()
target.close()

