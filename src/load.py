__doc__ = '''Utils for loading tweets (as strings) and labels from the ConLL files provided in the SentiMix Challenge
'''

import numpy as np


def detokenize_tweet(tokens):
    """ Take a list of tokens and return a tweet string
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


def load_data(training_file, bert=True):
    """ Takes a file object (not path string) for training_file

    If bert:
    returns (tweet_ids, tweets, sentiment_labels), where each tweet is a string, with no language tags included.

    If not bert:
    returns (tweet_ids, tweets, sentiment_labels), where each tweet is a [(token, tag), (token, tag), (token, tag)] list
    """
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
            if len(line.strip().split('\t')) == 2:
                token, lang = tuple(line.strip().split('\t'))
                if bert:
                    tweet.append(token)
                else:
                    tweet.append((token, lang))
        elif tweet:
            if bert:
                tweets.append(detokenize_tweet(tweet))
            else:
                tweets.append(tuple(tweet))
            tweet = []
    if tweet:
        if bert:
            tweets.append(detokenize_tweet(tweet))
        else:
            tweets.append(tuple(tweet))
    return np.array(tweet_ids), np.array(tweets), np.array(sentiments)
