import re
import ast

with open("phraseLevel_clf_result.txt", "r") as file:
    lines = file.readlines()
    predict_sum = 0
    total_sum = 0
    gold_NL_pred_L = 0
    gold_L_pred_NL = 0
    for k in range(0, 300, 3):
        predict_vec = lines[k].strip()
        gold_vec = lines[k+1].strip()
        count = lines[k+2].strip()
        m = re.match(r'.*predicted (\d+) / total (\d+)', count)
        if m:
            predict_sent = int(m.group(1))
            all_sent = int(m.group(2))
            predict_sum += predict_sent
            total_sum += all_sent
            if predict_sent != all_sent:
                predict_vec = ast.literal_eval(predict_vec.rstrip(", device='cuda:0')").lstrip("tensor("))
                gold_vec = ast.literal_eval(gold_vec.rstrip(", device='cuda:0')").lstrip("tensor("))
                for i in range(0, len(predict_vec)):
                    if predict_vec[i] == 0 and gold_vec[i] == 1:
                        gold_NL_pred_L += 1
                    elif predict_vec[i] == 1 and gold_vec[i] == 0:
                        gold_L_pred_NL += 1
                sent_id = int(k/3)+1
                ## detailed analysis
                print(sent_id, predict_vec, gold_vec, predict_sent, "/", all_sent)
    # print(gold_L_pred_NL)
    # print(gold_NL_pred_L)
    # print(predict_sum, total_sum)   # 300/352 85.2%
