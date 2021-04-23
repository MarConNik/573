import argparse
import joblib

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier


'''
python train.py --train-file data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll --model-file model.joblib
'''


def load_train_data(training_file):
    # FIXME: This is kind of janky; ideally we could use a CONLL parser
    tweets = []
    tweet_ids = []
    sentiments = []
    tweet = []
    for line in training_file:
        if line.strip() and line.split()[0] == 'meta' and not tweet:
            _, tweet_id, sentiment = line.strip().split('\t')
            tweet_ids.append(tweet_id)
            sentiments.append(sentiment)
        elif line.strip():
            token, lang = tuple(line.strip().split('\t'))
            tweet.append((token, lang))
        elif tweet:
            tweets.append(tuple(tweet))
            tweet = []
    if tweet:
        tweets.append(tuple(tweet))
    return np.array(tweet_ids), np.array(tweets), np.array(sentiments)


if __name__ == '__main__':
    # Parser arguments:
    parser = argparse.ArgumentParser(description='classify records in a test set')
    parser.add_argument('--model-file', type=argparse.FileType('wb'),
                        help='path to save model to')
    parser.add_argument('--train-file',
                        type=argparse.FileType('r', encoding='latin-1'),
                        help='path to unlabelled testing data file')
    args = parser.parse_args()

    tweet_ids, tweets, tweet_sentiments = load_train_data(args.train_file)

    vectorizer = CountVectorizer(
        ngram_range=(1, 3)
    )
    model = SGDClassifier()
