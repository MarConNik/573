#!/bin/sh

# Variables: 
# $1: training file
# $2: Model output file
# $3: Test file 
# $4: Gold labeled file
# $5: Result files
echo "Training model now."
python train.py --train-file $1 --model-file $2 
echo "Model training finished."

echo "Now classifying testing instances using model.."
python classify.py --test-file $3 --model-file $2 --output-file $3
echo "Finished classifying testing instances."

echo "Evaluate script running..."
python evaluate.py $4 $3 $5 
echo "Finished evaluating. 
echo "Results available here:"
echo $5