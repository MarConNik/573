MODEL_DIR="$OUTPUTS_DIR/${LANG_PAIR}-eval-model"
OUTPUT_FILE="$OUTPUTS_DIR/${LANG_PAIR}-eval.results"
EVAL_FILE="$RESULTS_DIR/${LANG_PAIR}-eval.txt"

echo "Running eval for ${LANG_PAIR}"

# Train, classify, and evaluate on Eval set
echo "training..."
python src/train.py --train-file "$TRAIN_FILE" --dev-file $DEV_FILE --model-directory "$MODEL_DIR"
echo "classifying..."
python src/classify.py --model-directory "${MODEL_DIR}/FINAL" --test-file "$TEST_FILE" --output-file $OUTPUT_FILE
if [[ "$LANG_PAIR" == "hinglish" ]]
then
  echo "evaluating..."
  python src/evaluate.py --train-file "$GOLD_STANDARD" "$OUTPUT_FILE" > "$EVAL_FILE"
fi
echo "complete!"
