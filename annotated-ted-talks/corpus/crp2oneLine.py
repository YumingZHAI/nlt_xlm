# transform file.crp to file.txt, where each aligned pair of sentences is written in one line
# and separated by '#'
# be sure that the ordrer is the same with the corresponding .aln file
# usage: e.g. python3 crp2oneLine.py subCorpus1.crp subCorpus1.txt
import os, sys, re   

source = sys.argv[1]
output = open(sys.argv[2], "w")

# leave the first line empty, to be iterated simultaneously with file.aln
output.write("\n")

with open(os.path.abspath(source)) as corpus:
	for line in corpus:
		if re.match(r'\d+\n', line):
			output.write(next(corpus).strip("\n") + "#" + next(corpus))

output.close()

