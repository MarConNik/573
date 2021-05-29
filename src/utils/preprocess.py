from nltk import tokenize

from emoji_translate.emoji_translate import Translator

emo = Translator(exact_match_only=False, randomize=True)
tweet_tokenizer = tokenize.TweetTokenizer()


def translate_emoji(tok):
    new_tok = emo.demojify(tok)
    if new_tok != tok:
        new_tok = emo.demojify(' '.join(tok)).split(' ')
    return new_tok

def detokenize_mentions_url(tokens, tags):
    """ Take a list of tokens (in Spanglish or Hinglish format) and tags and
    return a modified list of tokens combining urls and mentions, while regrouping tags
    (e.g. ["@", "connor", ",", "how", "are", "you", "?"] -> ["@connor" "," ["how"] ["are"] ["you"], ["?"])
    (tags: ["0", "Eng", "0", "Eng", "Eng", "Eng", "0"] -> [["0", "Eng"], ["0"], ["Eng"], ["Eng"], ["Eng"], ["0"]] )
    """
    tweet = []
    grouped_tags = []
    current_token = ''
    prev_token = ''
    prev_prev_token = ''
    for i, token in enumerate(tokens):
        #print(f'token: {token} tag: {tags[i]} tweet: {tweet} grouped: {grouped_tags}')
        if prev_token == '' and prev_prev_token == '':
            tweet.append(token)
            grouped_tags.append([tags[i]])
        elif token == '//' and prev_token in ('https', 'http'):
            tweet[-1] += ':' + token
            grouped_tags[-1].append(tags[i])
        elif prev_token == '//' and token == 't':
            tweet[-1] += token
            grouped_tags[-1].append(tags[i])
        elif token == 'co' and prev_token == '.':
            tweet[-1] += token
            grouped_tags[-1].append(tags[i])
        elif token == 'co' and prev_token == 't':
            tweet[-1] += '.' + token
            grouped_tags[-1].append(tags[i])
        elif prev_prev_token == 'co' and prev_token == '/':
            tweet[-1] += token
            grouped_tags[-1].append(tags[i])
        elif token and token[0] == '/':
            tweet[-1] += token
            grouped_tags[-1].append(tags[i])
        elif prev_token == '@':
            tweet[-1] += token
            grouped_tags[-1].append(tags[i])
        # I think we can keep the punctuation as a separate token
        # elif token and token[0] in {',', '.', '!', '?'}:
        #     current_token += token
        #     current_tag_group.append(tags[i])
        else:
            tweet.append(token)
            grouped_tags.append([tags[i]])
        prev_prev_token = prev_token
        prev_token = token
    return tweet, grouped_tags


def preprocess_token(token):
    '''Preprocess a token (as produced by the NLTK tweet tokenizer) in a tweet; replace URLs and twitter handles with
    generic token
    '''
    if token.startswith('@'):
        return '@USER'
    elif token.startswith('https://') or token.startswith('http://'):
        return 'HTTPURL'
    else:
        return translate_emoji(token)

def flatten_list(tokens, tag_groups):
    new_list = []
    new_tags = []
    for i, token in enumerate(tokens):
        if type(token) is list:
            for emoji in token:
                new_list.append(emoji)
                new_tags.append(tag_groups[i])
        else:
            new_list.append(token)
            new_tags.append(tag_groups[i])
    return(new_list, new_tags)

def preprocess_tweet(tweet, tags):
    '''Preprocess a tweet string by running NLTK tweet tokenizer on it, then filtering those tokens for
    '''
    tokens, tag_groups = detokenize_mentions_url(tweet, tags)
    processed_tokens = [preprocess_token(token) for token in tokens]
    flattened_tokens, tag_groups = flatten_list(processed_tokens, tag_groups)
    return flattened_tokens
