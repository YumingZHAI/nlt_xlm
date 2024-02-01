# this script calculates the standard deviation (écart-type) given a list/table of values

# input is a table of real values, for example:
# donnees = [0.59550562,0.54887218,0.56603774,0.54339623,0.59469697]

def moyenne(tableau):
    return sum(tableau, 0.0) / len(tableau)
# print(moyenne(donnees))

# La variance est définie comme la moyenne des carrés des écarts à la moyenne:
def variance(tableau):
    m = moyenne(tableau)
    return moyenne([(x-m)**2 for x in tableau])
# print(variance(donnees))

# L'écart-type est défini comme la racine carrée de la variance:
# écart-type = standard deviation (English term)
def ecartype(tableau):
    return variance(tableau)**0.5
# print(ecartype(donnees))

# train and dev europarl, 10 folds
# a = [84.6, 84.1, 84.7, 83.8, 86.0, 84.4, 84.7, 84.8, 84.2, 84.2]
# train and dev opensubtitles, 10 folds
# a = [86.8, 69.5, 71.7, 82.7, 87.3, 76.1, 76.7, 79.8, 78.5, 79.5]
# train and dev tedtalks, 10 folds
# a = [72.9, 74.4, 74.2, 75.6, 77.4, 71.6, 72.4, 74.1, 73.5, 73.1]
# train and dev books, 10 folds
# a = [97.6, 95.5, 95.9, 92.7, 93.3, 95.4, 94.5, 92.5, 92.9, 92.3]

# train europarl, dev opensubtitles
# a = [81.8, 66.8, 66.3, 76.4, 84.6, 68.8, 69.0, 71.8, 70.2, 69.8]
# train europarl, dev tedtalks
# a = [66.3, 68.7, 64.9, 71.2, 69.0, 63.9, 63.4, 63.9, 66.9, 66.2]
# train europarl, dev books
a = [92.1, 91.3, 91.0, 85.9, 85.4, 89.6, 87.7, 85.7, 88.6, 87.0]
# train opensubtitles, dev europarl
# a = [70.0, 72.4, 68.8, 63.3, 64.5, 64.4, 65.8, 66.2, 73.5, 66.8]
# train opensubtitles, dev tedtalks
# a = [66.8, 65.3, 62.1, 65.3, 66.2, 62.2, 61.5, 64.2, 67.6, 64.2]
# train opensubtitles, dev books
# a = [91.7, 84.4, 86.4, 76.3, 74.3, 82.8, 84.1, 81.9, 87.0, 82.8]
# train tedtalks, dev europarl
# a = [72.1, 75.7, 78, 72.4, 73.6, 74.3, 76, 76.8, 76.9, 73]
# train tedtalks, dev opensubtitles
# a = [85.2, 66.2, 69.6, 76.6, 84.8, 70.6, 69.2, 69.2, 68.8, 68.2]
# train tedtalks, dev books
# a = [92.5, 86.5, 87, 84.4, 85.6, 88.8, 85.2, 83.4, 85.4, 85.1]
# train books, dev europarl
# a = [70.5, 72.7, 72.9, 69.4, 74.8, 66.5, 66.2, 73.1, 69.1, 76.6 ]
# train books, dev opensubtitles
# a = [83.5, 64.9, 62.7, 76.1, 84.8, 68.5, 66.6, 71.2, 70.5, 68.8]
# train books, dev tedtalks
# a = [60.5, 65.5, 64.6, 64.6, 64.8, 59.0, 58.4, 60.5, 60.5, 64.5]

# train and dev tedannote, 10 folds
# a = [85.5, 79.2, 82.7, 75.7, 75.6, 79.1, 76.2, 82.6, 71.5, 78.5]

# load final europarl, resume training on tedannote
# a = [85.5, 73.4, 82.1, 80.3, 75.6, 80.8, 78.5, 80.8, 77.9, 83.7]
# load final opensubtitles, resume training on tedannote
# a = [82.1, 72.3, 80.9, 80.3, 75, 79.1, 80.2, 77.9, 73.8, 79.1]
# load final tedtalks, resume training on tedannote
# a = [80.3, 76.3, 81.5, 82.7, 76.2, 76.7, 72.1, 77.9, 73.8, 84.9]
# load final books, resume training on tedannote
# a = [80.3, 76.9, 84.4, 81.5, 77.3, 80.8, 78.5, 80.2, 73.3, 88.4]

# balanced phrase level clf, 10 folds train and dev
# a = [83.2, 81.5, 82.3, 83.5, 80.9, 83.7, 84.8, 82.7, 85.6, 85.3]

print(moyenne(a))
print(ecartype(a))

