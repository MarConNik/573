import argparse
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from transformers import AdamW, get_linear_schedule_with_warmup

from src.utils import Preprocessor, Tokenizer, load_train_data, BERT_MODEL_NAME
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification
from torch.utils.data import DataLoader


'''
E.g.:
python src/train.py --train-file data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll --model-file model.joblib
'''


# Default training hyperparameters
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_EPSILON = 1e-8
DEFAULT_NUM_EPOCHS = 4


def train_model(
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epsilon: float = DEFAULT_EPSILON,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    num_labels: int
) -> BertPreTrainedModel:
    model: BertPreTrainedModel = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels; -, 0, + for sentiment
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    # FIXME: Make sure your computer can run GPU
    # FIXME: Also, I (Connor) am pretty sure that since I have an AMD GPU, I have to activate something other than CUDA
    model.cuda()

    # FIXME: The tutorial that we are following says to use the HuggingFace/Transformers version; there is a PyTorch
    #  version now, though.
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        eps=epsilon
    )

    training_steps = len(training_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # FIXME: Why run with warmup if we set warmup steps to 0?
        num_training_steps=training_steps
    )




if __name__ == '__main__':
    # Parser arguments:
    parser = argparse.ArgumentParser(description='classify records in a test set')
    parser.add_argument('--model-file', type=argparse.FileType('wb'),
                        help='path to save model to')
    parser.add_argument('--train-file',
                        type=argparse.FileType('r', encoding='latin-1'),
                        help='path to unlabelled testing data file')
    parser.add_argument('--dev-file',
                        type=argparse.FileType('r', encoding='latin-1'),
                        help='path to unlabelled testing data file')
    args = parser.parse_args()

    # Get training tweets from file
    train_ids, train_tweets, train_sentiments = \
        load_train_data(args.train_file, False)

    # Transform tweets into sparse vectors
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        tokenizer=Tokenizer(),
        preprocessor=Preprocessor()
    )
    train_vectors = vectorizer.fit_transform(train_tweets)

    # Train model
    model = SGDClassifier()
    model.fit(train_vectors, train_sentiments)

    # Save model to joblib
    joblib.dump((vectorizer, model), args.model_file)

    # Run model on test data
    # dev_ids, dev_tweets, dev_sentiments = load_train_data(args.dev_file)
    # dev_vectors = vectorizer.transform(dev_tweets)
    # dev_predictions = model.predict(dev_vectors)
