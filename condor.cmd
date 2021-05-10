executable = src/run.sh
getenv     = true
notification = complete
transfer_executable = false
request_memory = 2*1024
request_GPUs = 1
arguments = "outputs/baseline/MODEL_NAME_HERE data/Semeval_2020_task9_data/Spanglish/Spanglish_dev.conll outputs/D3/model_predictions.txt data/Semeval_2020_task9_data/Spanglish/test/Spanglish_dev_gold_labels.conll results/D3_scores.OUT"
queue
