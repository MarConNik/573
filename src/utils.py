import numpy as np
import torch
from torch.utils.data import random_split, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from src.preprocess import preprocess_tweet

LABEL_INDICES = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}
INDEX_LABELS = {value: key for key, value in LABEL_INDICES.items()}
DEFAULT_TRAIN_SHARE = 0.90
MAX_TOKENIZED_TWEET_LENGTH = 140
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


def load_data(training_file, bert=True):
    """ Takes a file object (not path string) for training_file

    If bert:
    returns (tweet_ids, tweets, sentiment_labels), where each tweet is a [(token, tag), (token, tag), (token, tag)] list

    If not bert:
    returns (tweet_ids, tweets, sentiment_labels), where each tweet is a string, with no language tags included.
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


def encode_strings(strings, labels):
    '''Preprocess tweet strings; tokenize with BERT tokenizer; map tokens to IDs; pad token sequences; create attention masks
    '''
    input_ids = []
    atten_masks = []
    for s in strings:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        processed_string = preprocess_tweet(s)
        encoded_dict = tokenizer.encode_plus(
            processed_string,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_TOKENIZED_TWEET_LENGTH,  # Pad & truncate all sentences.
            truncation=True,
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt', )  # Return pytorch tensors.

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        atten_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(atten_masks, dim=0)

    # Convert label strings to integers
    all_labels = set(labels)
    label_ints = np.array([LABEL_INDICES[label] for label in labels])
    label_tensor = torch.tensor(label_ints)
    return input_ids, attention_masks, label_tensor


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
