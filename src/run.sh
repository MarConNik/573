#!/bin/sh

# Variables: 
# $1: Model dir
# $2: Test file 
# $3: Result file (Spanglish_dev.conll for D3)
# $4: Gold labeled file
# $5: Evaluation file

# Example usage:
# src/run.sh data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll model.joblib data/Semeval_2020_task9_data/Spanglish/Spanglish_dev.conll data/Semeval_2020_task9_data/Spanglish/test/Spanglish_dev_gold_labels.conll result.txt evaluation.txt

echo "Using pretrained M-BERT model from dir here:" 
echo "$1"
echo "Now classifying testing instances using model.."
#python src/classify.py --model-directory "$1" --test-file "$2" --output-file "$3"
echo "Finished classifying testing instances."

echo "Evaluate script running..."
#python src/evaluate.py "$4" "$3" > "$5"
echo "Finished evaluating."
echo "Results available here:"
echo "$3"
echo "Evaluation available here:"
echo "$5"