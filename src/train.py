import argparse
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from src.utils import Preprocessor, Tokenizer, load_train_data
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from torch.utils.data import DataLoader


'''
E.g.:
python src/train.py --train-file data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll --model-file model.joblib
'''


def train_model(training_dataloader: DataLoader, validation_dataloader: DataLoader) -> BertPreTrainedModel:
    raise NotImplementedError()


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
