#!/bin/bash

# Set up environment
. "/home2/tampakis/anaconda3/etc/profile.d/conda.sh"
conda activate 573

# Make output & results directories
OUTPUTS_DIR="outputs/D4/adaptation"
mkdir -p "$OUTPUTS_DIR"
RESULTS_DIR="results/D4/adaptation"
mkdir -p "$RESULTS_DIR"

# Hinglish settings
HINGLISH_TRAIN_FILE="data/Semeval_2020_task9_data/Hinglish/Hinglish_train_14k_split_conll.txt"
HINGLISH_DEV_FILE="data/Semeval_2020_task9_data/Hinglish/Hinglish_dev_3k_split_conll.txt"
HINGLISH_TEST_FILE="data/Semeval_2020_task9_data/Hinglish/Hinglish_test_unlabeled_conll_updated.txt"
HINGLISH_GOLD_STANDARD="data/Semeval_2020_task9_data/Hinglish/Hinglish_test_labels.txt"
HINGLISH_MODEL_DIR="$OUTPUTS_DIR/hinglish-model"
HINGLISH_OUTPUT_FILE="$OUTPUTS_DIR/hinglish-test.results"
HINGLISH_EVAL_FILE="$RESULTS_DIR/hinglish-eval.txt"

# Train, run, and evaulate Hinglish model (dev)
python src/train.py --train-file "$HINGLISH_TRAIN_FILE" --model-directory "$HINGLISH_MODEL_DIR"
python src/classify.py --model-directory "$HINGLISH_MODEL_DIR" --test-file "$HINGLISH_DEV_FILE" --output-file $HINGLISH_OUTPUT_FILE
python src/evaluate.py --train-file "$HINGLISH_GOLD_STANDARD" "$HINGLISH_OUTPUT_FILE" > "$HINGLISH_EVAL_FILE"

# Train, run, and evaluate Hinglish model (eval)
python src/train.py --train-file "$HINGLISH_TRAIN_FILE" --dev-file $HINGLISH_DEV_FILE --model-directory "$HINGLISH_MODEL_DIR"
python src/classify.py --model-directory "$HINGLISH_MODEL_DIR" --test-file "$HINGLISH_TEST_FILE" --output-file $HINGLISH_OUTPUT_FILE
python src/evaluate.py --train-file "$HINGLISH_GOLD_STANDARD" "$HINGLISH_OUTPUT_FILE" > "$HINGLISH_EVAL_FILE"
