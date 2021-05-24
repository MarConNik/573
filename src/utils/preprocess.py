from nltk import tokenize

from .load import detokenize_tweet
from emoji_translate.emoji_translate import Translator

emo = Translator(exact_match_only=False, randomize=True)
tweet_tokenizer = tokenize.TweetTokenizer()


def preprocess_token(token):
    '''Preprocess a token (as produced by the NLTK tweet tokenizer) in a tweet; replace URLs and twitter handles with
    generic token
    '''
    token = translate_emoji(token)
    if token.startswith('@'):
        return '@USER'
    elif token.startswith('https://') or token.startswith('http://'):
        return 'HTTPURL'
    else:
        return token

def translate_emoji(tok):
    new_tok = emo.demojify(tok)
    if new_tok != tok:
        new_tok = emo.demojify(' '.join(tok))
    return new_tok

def preprocess_tweet(tweet):
    '''Preprocess a tweet string by running NLTK tweet tokenizer on it, then filtering those tokens for
    '''
    tokens = tweet.split(' ')
    return detokenize_tweet([preprocess_token(token) for token in tokens])
