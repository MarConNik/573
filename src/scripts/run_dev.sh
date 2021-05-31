MODEL_DIR="$OUTPUTS_DIR/${LANG_PAIR}-dev-model"
OUTPUT_FILE="$OUTPUTS_DIR/${LANG_PAIR}-dev.results"
EVAL_FILE="$RESULTS_DIR/${LANG_PAIR}-dev.txt"

echo "Running dev for ${LANG_PAIR}"

# Train, classify, and evaluate on Dev dataset
echo "training..."
python src/train.py --train-file "$TRAIN_FILE" --model-directory "$MODEL_DIR"
echo "classifying..."
python src/classify.py --model-directory "${MODEL_DIR}/FINAL" --test-file "$TEST_FILE" --output-file $OUTPUT_FILE
echo "evaluating..."
python src/evaluate.py --train-file "$GOLD_STANDARD" "$OUTPUT_FILE" > "$EVAL_FILE"
echo "complete!"
