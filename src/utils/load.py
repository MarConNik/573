__doc__ = '''Utils for loading tweets (as strings) and labels from the ConLL files provided in the SentiMix Challenge
'''

import ftfy
import numpy as np


def load_lince_test_data(line, data_file):
    lines = [x.strip() for x in data_file.readlines()]
    lines.insert(0, line)
    tweets = []
    tags = []
    cur_tweet = []
    cur_tag_list = []
    for line in lines:
        if line == '':
            tweets.append(cur_tweet)
            tags.append(cur_tag_list)
            cur_tweet = []
            cur_tag_list = []
        else:
            tweet_toke = line.split('\t')[0]
            tweet_tag = line.split('\t')[1]
            cur_tweet.append(tweet_toke)
            cur_tag_list.append(tweet_tag)
    
    tweet_ids = [x for x in range(1,len(tweets)+1)]
    return tweet_ids, tweets, tags
   

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
    line_index = 0
    for line in data_file:
        line_index +=1
        if line_index == 1:
            # check first line for meta or sent_enum
            if 'meta' not in line and 'sent_enum' not in line:
                tweet_ids, tweets, tags = load_lince_test_data(line.strip(), data_file)
                return np.array(tweet_ids), np.array(tweets), np.array(tags) 

        # for train.conll, dev.conll
        if line.strip() and line.startswith('# sent_enum = ') and len(tweet) == 0:
            if with_labels:
                id_sentiment = line.split('# sent_enum = ')[1]
                tweet_id = id_sentiment.split()[0]
                sentiment = id_sentiment.split()[1]
                sentiments.append(sentiment)
            else:
                _, tweet_id = line.strip().split('\t')[:2]
            tweet_ids.append(tweet_id)
        
        # for Hinglish train, dev, test files
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
        
        
    # tweet_ids, tweets, tags are returned on line 53 for test.conll because there is no meta /sent_enum
    if with_labels:
        return np.array(tweet_ids), np.array(tweets), np.array(tags), np.array(sentiments)
    else:
        return np.array(tweet_ids), np.array(tweets), np.array(tags)
