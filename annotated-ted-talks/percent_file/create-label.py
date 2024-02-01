label = open("all.label", "w")
# absent: 0: don't contain non-literal translation
# present: 1: contain non-literal translation

with open("all_percent.txt", "r") as file:
    # all_literal = 0
    # contain_n_literal = 0
    for l in file:
        if l.strip() == "0.0":
            # all_literal += 1
            print("absent", file=label)
        else:
            # contain_n_literal += 1
            print("present", file=label)
# print(all_literal)
# print(contain_n_literal)
label.close()