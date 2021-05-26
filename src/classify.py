import argparse
import os
import numpy as np
import csv
from utils.load import load_data
from utils.bert import INDEX_LABELS, encode_strings
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.model import MODEL_FILENAME, BertLSTMClassifier

'''E.g.:
python src/classify.py --model-directory outputs/SpanglishModel-V0/ --test-file data/Semeval_2020_task9_data/Spanglish/Spanglish_test_conll_unlabeled.txt --output-file Spanglish_predictions.txt
'''


DEFAULT_BATCH_SIZE = 32


def output_predictions(X_ids, z, output_file):
    """ Takes a list of ids along with classified sentiments and writes to provided output filename.
    """
    filename = output_file.name
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Uid', 'Sentiment'])
        for i in range(len(X_ids)):
            X_id = X_ids[i].replace('\n','').replace('\r','')
            guess = z[i].replace('\n','').replace('\r','')
            writer.writerow([X_id, guess])


if __name__ == '__main__':
    # Parser arguments:
    parser = argparse.ArgumentParser(description='classify records in a test set')
    parser.add_argument('--model-directory', '--model-dir', type=str,
                        help='path to saved model to use in classification')
    parser.add_argument('--test-file',
                        type=argparse.FileType('r', encoding='latin-1'),
                        help='path to unlabelled testing data file')
    parser.add_argument('--output-file', type=argparse.FileType('w'),
                        help='path to output file')
    parser.add_argument('--batch-size', type=int,
                        default=DEFAULT_BATCH_SIZE,
                        help='training (and validation) batch size')
    args = parser.parse_args()

    # Initialize model architecture
    model = BertLSTMClassifier(num_labels=len(INDEX_LABELS))
    # Load pre-saved model parameters
    model_path = os.path.join(args.model_directory, MODEL_FILENAME)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)

    # Set model to evaluate mode
    model.eval()

    # Use CUDA if it's available (ie on Google Colab)
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")
        model.to(device)

    # Load test data
    X_ids, X = load_data(args.test_file, with_labels=False)
    input_ids, attention_masks, _ = encode_strings(X, [])
    dataset = TensorDataset(input_ids, attention_masks)

    prediction_dataset = DataLoader(dataset, batch_size = args.batch_size)

    predictions = []
    for batch in prediction_dataset:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # For some reason we have to globally disable gradient accumulation (in addition to setting model to `eval()`)
        with torch.no_grad(): result = model(b_input_ids,
                         token_type_ids=None,
                         attention_mask=b_input_mask,
                         return_dict=True)
        batch_logits = result.logits
        logits = batch_logits.detach().cpu().numpy()
        predictions += [INDEX_LABELS[np.argmax(values)] for values in logits]

    output_predictions(X_ids, predictions, args.output_file)
