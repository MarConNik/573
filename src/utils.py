import numpy as np


class Preprocessor:
    def __init__(self):
        pass

    def __call__(self, token):
        # TODO: Come up with preprocessing
        return token


class Tokenizer:
    def __init__(self):
        pass

    def __call__(self, sequence):
        '''Take as input sequence [(token1, langtag1), (token2, langtag2), ...], output shallow sequence:
        [token1, token2, ..., '', '', '', langtag1, langtag2, ...]
        '''
        return [token for token, language in sequence] + ([''] * 3) + [language for token, language in sequence]


def load_train_data(training_file, bert=True):
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
            if len(line.strip().split('\t')) == 2:
                token, lang = tuple(line.strip().split('\t'))
                if bert:
                    tweet.append(token)
                else:
                    tweet.append((token, lang))
        elif tweet:
            if bert:
                tweets.append(' '.join(tweet))
            else:
                tweets.append(tuple(tweet))
            tweet = []
    if tweet:
        if bert:
            tweets.append(' '.join(tweet))
        else:
            tweets.append(tuple(tweet))
    return np.array(tweet_ids), np.array(tweets), np.array(sentiments)
