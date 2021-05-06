import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

DEFAULT_TRAIN_SHARE = 0.90


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


def get_dataloaders(input_ids, attention_masks, sentiment_labels, batch_size: int, train_share=DEFAULT_TRAIN_SHARE):
    # Split dataset into training and validation subsets
    dataset = TensorDataset(input_ids, attention_masks, sentiment_labels)
    training_size = int(train_share * len(dataset))
    validation_size = len(dataset) - training_size
    training_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])

    # Create data loaders; training data is loaded in random order
    # NOTE: "validation" should be a subset of what was given in the "training" file; it has nothing to do with the
    # contents of the "dev" file, which is used for intermediate model evaluation
    training_dataloader = DataLoader(
        training_dataset,
        sampler=RandomSampler(training_dataset),
        batch_size=batch_size
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        sampler=SequentialSampler(validation_dataset),
        batch_size=batch_size
    )

    return training_dataloader, validation_dataloader
