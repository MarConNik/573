DEV_OUTPUTS_DIR="$OUTPUTS_DIR/devtest"
mkdir -p $DEV_OUTPUTS_DIR
DEV_RESULTS_DIR="$RESULTS_DIR/devtest"
mkdir -p $DEV_RESULTS_DIR

MODEL_DIR="$DEV_OUTPUTS_DIR/${LANG_PAIR}-dev-model"
OUTPUT_FILE="$DEV_OUTPUTS_DIR/${LANG_PAIR}-dev.results"
EVAL_FILE="$DEV_RESULTS_DIR/D4_scores.out"

echo "Running dev for ${LANG_PAIR}"

# Train, classify, and evaluate on Dev dataset
echo "training..."
python src/train.py --train-file "$TRAIN_FILE" --model-directory "$MODEL_DIR"
echo "classifying..."
python src/classify.py --model-directory "${MODEL_DIR}/FINAL" --test-file "$TEST_FILE" --output-file $OUTPUT_FILE
echo "evaluating..."
python src/evaluate.py --train-file "$GOLD_STANDARD" "$OUTPUT_FILE" > "$EVAL_FILE"
echo "complete!"
