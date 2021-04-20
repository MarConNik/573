import argparse
import numpy as np
import csv
from sklearn.dummy import DummyClassifier


def load_test_data(test_file):
    # FIXME: This is kind of janky; ideally we could use a CONLL parser
    tweets = []
    tweet_ids = []
    tweet = []
    for line in test_file:
        if line.strip() and line.split()[0] == 'meta' and not tweet:
            tweet_ids.append(line.strip().split('\t')[1])
        elif line.strip():
            tweet.append(tuple(line.strip().split('\t')))
        elif tweet:
            tweets.append(tuple(tweet))
            tweet = []
    if tweet:
        tweets.append(tuple(tweet))
    return np.array(tweet_ids), np.array(tweets)


def output_predictions(X_ids, z, output_file):
    output_csv = csv.writer(output_file, quoting=csv.QUOTE_MINIMAL)
    output_csv.writerow(['Uid', 'Sentiment'])
    for i in range(len(X_ids)):
        X_id = X_ids[i]
        guess = z[i]
        output_csv.writerow([X_id, guess])


if __name__ == '__main__':
    # Parser arguments:
    parser = argparse.ArgumentParser(description='classify records in a test set')
    parser.add_argument('--model-file', type=argparse.FileType('rb'),
                        help='path to saved model to use in classification')
    parser.add_argument('--test-file',
                        type=argparse.FileType('r', encoding='latin-1'),
                        help='path to unlabelled testing data file')
    parser.add_argument('--output-file', type=argparse.FileType('w'),
                        help='path to output file')
    args = parser.parse_args()

    # Load test data
    X_ids, X = load_test_data(args.test_file)

    # TODO: load real model from file
    classifier = DummyClassifier(strategy='constant', constant='positive')
    classifier.fit([''], ['positive'])
    z = classifier.predict(X)

    output_predictions(X_ids, z, args.output_file)