# python3 calculate_f1.py gold_file model_output > results.txt

import sys

def calc_f1(precision, recall):
    return(100*2*precision*recall/(precision+recall))

gold_file_path = sys.argv[1]
model_output_path = sys.argv[2]


confusion = {}
confusion['positive'] = { 'positive': 0, 'neutral': 0, 'negative': 0 }
confusion['neutral']  = { 'positive': 0, 'neutral': 0, 'negative': 0 }
confusion['negative']  = { 'positive': 0, 'neutral': 0, 'negative': 0 }

gold_results = {}
gold_file = open(gold_file_path, "r")
gold_lines = gold_file.readlines()
for line in gold_lines:
    if line == "Uid,Sentiment\n":
        continue
    else:
        id,sentiment = line.strip().split(',')
        gold_results[id] = sentiment

model_output = open(model_output_path, "r")
model_lines = model_output.readlines()
for line in model_lines:
    if line == "Uid,Sentiment\n":
        continue
    else:
        id,sentiment = line.strip().split(',')
        correct_sentiment = gold_results[id]
        confusion[correct_sentiment][sentiment] += 1

pos_count = sum(confusion['positive'].values())
neu_count = sum(confusion['neutral'].values())
neg_count = sum(confusion['negative'].values())
total_count = pos_count + neu_count + neg_count

pos_predicted = confusion['positive']['positive'] + confusion['neutral']['positive'] + confusion['negative']['positive']
neu_predicted = confusion['positive']['neutral'] + confusion['neutral']['neutral'] + confusion['negative']['neutral']
neg_predicted = confusion['positive']['negative'] + confusion['neutral']['negative'] + confusion['negative']['negative']

pos_recall = confusion['positive']['positive']/pos_count
neu_recall = confusion['neutral']['neutral']/neu_count
neg_recall = confusion['negative']['negative']/neg_count

pos_precision = confusion['positive']['positive']/pos_predicted
neu_precision = confusion['neutral']['neutral']/neu_predicted
neg_precision = confusion['negative']['negative']/neg_predicted

f1_pos = calc_f1(pos_precision, pos_recall)
f1_neu = calc_f1(neu_precision, neu_recall)
f1_neg = calc_f1(neg_precision, neg_recall)

f1_pos_weighted = f1_pos * pos_count / total_count
f1_neu_weighted = f1_neu * neu_count / total_count
f1_neg_weighted = f1_neg * neg_count / total_count
f1_weighted = f1_pos_weighted + f1_neu_weighted + f1_neg_weighted

print(f"Weighted F1 score: {f1_weighted}")
print(f"Positive: Precision={pos_precision} Recall={pos_recall} F1={f1_pos}")
print(f"Neutral: Precision={neu_precision} Recall={neu_recall} F1={f1_neu}")
print(f"Negative: Precision={neg_precision} Recall={neg_recall} F1={f1_neg}")
print()


print("Confusion matrix for the output data:")
print("row is the truth, column is the system output")
sys.stdout.write("       ")
for cat in confusion.keys():
    sys.stdout.write(cat.ljust(9))
sys.stdout.write("\n")

for row in confusion.keys():
    sys.stdout.write(row.ljust(9))
    for col, cnt in confusion[row].items():
        sys.stdout.write(f' {str(cnt).ljust(8)}')
    sys.stdout.write("\n")
