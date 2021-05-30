__doc__ = '''Utils for loading tweets (as strings) and labels from the ConLL files provided in the SentiMix Challenge
'''

import ftfy
import numpy as np


def load_data(data_file, with_labels=True):
    """ Takes a file object (not path string) for data_file

    If with_labels:
        returns (tweet_ids, tweets, tags, sentiment_labels), where each tweet is a list of tokens, with language tags included.

    Else:
        returns (tweet_ids, tweets, tags), where each tweet is a list of tokens, with language tags included.
    """
    tweets = []
    tags = []
    tweet_ids = []
    sentiments = []
    tweet = []
    tweet_tags = []
    for line in data_file:
        
        if line.strip() and line.startswith('# sent_enum = ') and len(tweet) == 0:
            if with_labels:
                id_sentiment = line.split('# sent_enum = ')[1]
                tweet_id = id_sentiment.split()[0]
                sentiment = id_sentiment.split()[1]
                sentiments.append(sentiment)
            else:
                _, tweet_id = line.strip().split('\t')[:2]
            tweet_ids.append(tweet_id)
        
        elif line.strip() and line.split()[0] == 'meta' and len(tweet) == 0:
            if with_labels:
                _, tweet_id, sentiment = line.strip().split('\t')
                sentiments.append(sentiment)
            else:
                _, tweet_id = line.strip().split('\t')[:2]
            tweet_ids.append(tweet_id)
            
            
        elif line.strip():
            if len(line.strip().split('\t')) == 2:
                token, lang = tuple(line.strip().split('\t'))
                tweet_tags.append(lang)
                tweet.append(ftfy.fix_text(token))
        elif tweet:
            tweets.append(tweet)
            tweet = []
            tags.append(tweet_tags)
            tweet_tags = []
    if tweet:
        tweets.append(tweet)
        tags.append(tweet_tags)
        
    if with_labels:
        return np.array(tweet_ids), np.array(tweets), np.array(tags), np.array(sentiments)
    else:
        return np.array(tweet_ids), np.array(tweets), np.array(tags)
