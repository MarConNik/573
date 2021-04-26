#!/bin/sh

# Variables: 
# $1: Training file
# $2: Model file
# $3: Test file 
# $4: Gold labeled file
# $5: Result file
# $6: Evaluation file

# Example usage:
# src/run.sh data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll model.joblib data/Semeval_2020_task9_data/Spanglish/Spanglish_dev.conll data/Semeval_2020_task9_data/Spanglish/test/Spanglish_dev_gold_labels.conll result.txt evaluation.txt

echo "Training model now."
python src/train.py --train-file "$1" --model-file "$2"
echo "Model training finished."

echo "Now classifying testing instances using model.."
python src/classify.py --test-file "$3" --model-file "$2" --output-file "$5"
echo "Finished classifying testing instances."

echo "Evaluate script running..."
python src/evaluate.py "$5" "$4" > "$6"
echo "Finished evaluating."
echo "Results available here:"
echo "$5"
echo "Evaluation available here:"
echo "$6"