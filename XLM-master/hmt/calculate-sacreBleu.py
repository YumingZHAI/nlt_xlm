import sacrebleu
# the input sentences for sacrebleu should be detokenized
# human = [["Reprise de la session Je déclare reprise la session du Parlement européen, qui avait été interrompue le vendredi 17 décembre 1999, et je vous souhaite à nouveau une bonne année, dans l'espoir que vous avez pu profiter d'une agréable période de fête.",
#          "Le paragraphe 6 du rapport Cunha sur les programmes d'orientation pluriannuels, qui sera soumis au Parlement ce jeudi, propose d'introduire des sanctions applicables aux pays qui ne respectent pas les objectifs annuels de réduction de leur flotte.",
#          "Je voudrais savoir si l'on peut avancer une objection de ce type à ce qui n'est qu'un rapport, pas une proposition législative, et si je suis habilité à le faire ce jeudi."]]
# machine = ["Reprise de la session Je déclare reprise la session du Parlement européen qui avait été interrompue le vendredi 17 décembre dernier et je vous renouvelle tous mes vux en espérant que vous avez passé de bonnes vacances.",
#          "Le rapport Cunha sur les programmes d'orientation pluriannuels est présenté devant le Parlement jeudi et contient une proposition au paragraphe 6 visant à introduire une forme de pénalités sous forme de quotas pour les pays qui ne respectent pas leurs objectifs annuels de réduction de la flotte.",
#          "Je voudrais savoir si l'on peut soulever une objection de ce genre à ce qui n'est qu'un rapport, et non une proposition législative, et si je peux le faire de manière compétente jeudi."]

# data/orig/top30k-tedtalks-fairseq.fr
# data/orig/top30k-tedtalks-human.fr

with open("literary_corpus/en-fr/11-books-fairseq.fr") as m, open("literary_corpus/en-fr/11_books_fr.txt") as h:
    human_list = []
    machine = []
    for l_m, l_h in zip(m, h):
        # print(l_m, l_h)
        human_list.append(l_h.strip())
        machine.append(l_m.strip())
        # break
    human = []
    human.append(human_list)

    print(len(human_list))
    print(len(machine))

bleu = sacrebleu.corpus_bleu(machine, human)
print(bleu.score)


