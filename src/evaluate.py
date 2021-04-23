# python3 calculate_f1.py gold_file model_output > results.txt

import sys
from sklearn.metrics import f1_score, precision_recall_fscore_support

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

model_results = {}
model_output = open(model_output_path, "r")
model_lines = model_output.readlines()
for line in model_lines:
    if line == "Uid,Sentiment\n":
        continue
    else:
        id,sentiment = line.strip().split(',')
        model_results[id] = sentiment
        correct_sentiment = gold_results[id]
        confusion[correct_sentiment][sentiment] += 1

gold_predictions = [i[1] for i in sorted(gold_results.items())]
model_predictions = [i[1] for i in sorted(model_results.items())]

f1_weighted = f1_score(gold_predictions, model_predictions, average='weighted')
precision, recall, f1, _ = precision_recall_fscore_support(gold_predictions,model_predictions,labels=['positive', 'neutral', 'negative'])
pos_precision, neu_precision, neg_precision = precision
pos_recall, neu_recall, neg_recall = recall
pos_f1, neu_f1, neg_f1 = f1


print(f"Weighted F1 score: {f1_weighted}")
print(f"Positive: Precision={pos_precision} Recall={pos_recall} F1={pos_f1}")
print(f"Neutral: Precision={neu_precision} Recall={neu_recall} F1={neu_f1}")
print(f"Negative: Precision={neg_precision} Recall={neg_recall} F1={neg_f1}")
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
