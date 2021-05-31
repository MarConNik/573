#!/bin/bash

# Set up environment
. "/home2/tampakis/anaconda3/etc/profile.d/conda.sh"
conda activate 573

# Make output & results directories
HINGLISH_OUTPUTS_DIR="outputs/D4/adaptation"
mkdir -p "$HINGLISH_OUTPUTS_DIR"
HINGLISH_RESULTS_DIR="results/D4/adaptation"
mkdir -p "$HINGLISH_RESULTS_DIR"

# Hinglish settings
HINGLISH_TRAIN_FILE="data/Semeval_2020_task9_data/Hinglish/Hinglish_train_14k_split_conll.txt"
HINGLISH_DEV_FILE="data/Semeval_2020_task9_data/Hinglish/Hinglish_dev_3k_split_conll.txt"
HINGLISH_TEST_FILE="data/Semeval_2020_task9_data/Hinglish/Hinglish_test_unlabeled_conll_updated.txt"
HINGLISH_GOLD_STANDARD="data/Semeval_2020_task9_data/Hinglish/Hinglish_test_labels.txt"

# Train, run, and evaluate Hinglish model on dev
TRAIN_FILE="$HINGLISH_TRAIN_FILE" TEST_FILE="$HINGLISH_DEV_FILE" GOLD_STANDARD="$HINGLISH_GOLD_STANDARD" \
MODEL_DIR="$HINGLISH_MODEL_DIR" LANG_PAIR="hinglish" OUTPUTS_DIR="$HINGLISH_OUTPUTS_DIR" RESULTS_DIR="$HINGLISH_RESULTS_DIR" \
 src/scripts/run_dev.sh