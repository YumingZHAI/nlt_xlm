# cf Desktop/COLING2020/英法文学句对齐语料.txt 

with open("Books.en-fr.en") as en:
	en_lines = en.readlines() 
with open("Books.en-fr.fr") as fr:
	fr_lines = fr.readlines() 

name = "the_fall_of_the_house_of_usher"

output = open(name + "_en.txt", "w")
for x in range(69697, 69974):
	print(en_lines[x].strip(), file=output)
output.close() 

output = open(name + "_fr.txt", "w")
for x in range(69697, 69974):
	print(fr_lines[x].strip(), file=output)
output.close() 

# cat pride_and_prejudice_en.txt jane_eyre_en.txt alice\'s_adventures_in_wonderland_en.txt moll_flanders_en.txt robinson_crusoe_en.txt a_study_in_scarlet_en.txt the_great_shadow_en.txt the_hound_of_the_baskervilles_en.txt rodney_stone_en.txt three_men_in_a_boat_en.txt the_fall_of_the_house_of_usher_en.txt > 11_books_en.txt
# cat pride_and_prejudice_fr.txt jane_eyre_fr.txt alice\'s_adventures_in_wonderland_fr.txt moll_flanders_fr.txt robinson_crusoe_fr.txt a_study_in_scarlet_fr.txt the_great_shadow_fr.txt the_hound_of_the_baskervilles_fr.txt rodney_stone_fr.txt three_men_in_a_boat_fr.txt the_fall_of_the_house_of_usher_fr.txt > 11_books_fr.txt