executable = src/run.sh
getenv     = true
notification = complete
transfer_executable = false
request_memory = 2*1024
arguments = "data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll model.joblib data/Semeval_2020_task9_data/Spanglish/Spanglish_dev.conll data/Semeval_2020_task9_data/Spanglish/test/Spanglish_dev_gold_labels.conll result.txt evaluation.txt"
queue
