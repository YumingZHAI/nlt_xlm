#$PWD=percent_file/

for name in control-corpus subCorpus1 subCorpus2 subCorpus3 subCorpus4 subCorpus5 \
   subCorpus6 subCorpus7 subCorpus8 subCorpus9 subCorpus13 subCorpus14 subCorpus15 ; do
   python3 nonL-percent.py ${name}    # the ".txt" part is in the script
   cat ${name}.txt >> all_percent.txt
done
