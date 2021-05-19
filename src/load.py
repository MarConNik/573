__doc__ = '''Utils for loading tweets (as strings) and labels from the ConLL files provided in the SentiMix Challenge
'''

import numpy as np


def detokenize_tweet(tokens):
    """ Take a list of tokens (in Spanglish or Hinglish format) and return a tweet string
    (e.g. ["@", "connor", ",", "how", "are", "you", "?"] -> "@connor, how are you?")
    """
    tweet = ''
    prev_token = ''
    prev_prev_token = ''
    for token in tokens:
        if prev_token == '' and prev_prev_token == '':
            tweet += token
        elif token == '//' and prev_token == 'https':
            tweet += ':' + token
        elif prev_token == '//' and token == 't':
            tweet += token
        elif token == 'co' and prev_token == '.':
            tweet += token
        elif token == 'co' and prev_token == 't':
            tweet += '.' + token
        elif prev_prev_token == 'co' and prev_token == '/':
            tweet += token
        elif token and token[0] == '/':
            tweet += token
        elif prev_token == '@':
            tweet += token
        elif token and token[0] in {',', '.', '!', '?'}:
            tweet += token
        else:
            tweet += ' ' + token
        prev_prev_token = prev_token
        prev_token = token
    return tweet


def load_data(data_file, with_labels=True):
    """ Takes a file object (not path string) for data_file

    If with_labels:
        returns (tweet_ids, tweets, sentiment_labels), where each tweet is a string, with no language tags included.

    Else:
        returns (tweet_ids, tweets), where each tweet is a string, with no language tags included.
    """
    tweets = []
    tweet_ids = []
    sentiments = []
    tweet = []
    for line in data_file:
        if line.strip() and line.split()[0] == 'meta' and len(tweet) == 0:
            if with_labels:
                _, tweet_id, sentiment = line.strip().split('\t')
                sentiments.append(sentiment)
            else:
                _, tweet_id = line.strip().split('\t')[:2]
            tweet_ids.append(tweet_id)
        elif line.strip():
            if len(line.strip().split('\t')) == 2:
                token, lang = tuple(line.strip().split('\t'))
                tweet.append(token)
        elif tweet:
            tweets.append(detokenize_tweet(tweet))
            tweet = []
    if tweet:
        tweets.append(detokenize_tweet(tweet))
    if with_labels:
        return np.array(tweet_ids), np.array(tweets), np.array(sentiments)
    else:
        return np.array(tweet_ids), np.array(tweets)
