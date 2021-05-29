import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast

from .preprocess import preprocess_tweet

LABEL_INDICES = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}
MAP_TAGS = {
    'lang1': 0, # English
    'lang2': 1, # Spanglish
    'ne': 2, # named entity
    'ambiguous': 3, # could be Spanish or English
    'unk': 4, # unknown language
    'other': 5, # punctuation, emojis
    'mixed': 6, # both English and Spanish
    'fw': 7, # foreign word (not English or Spanish)
    'Eng': 0, # English
    'O': 5, # punctuation
    'Hin': 8, # Hindi
    'EMT': 5 # emojis
}
TAG_FEATURES = len(set(MAP_TAGS.values()))
INDEX_LABELS = {value: key for key, value in LABEL_INDICES.items()}
DEFAULT_TRAIN_SHARE = 0.90
MAX_TOKENIZED_TWEET_LENGTH = 140
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

def encode_strings(tokens, labels, tags):
    '''Preprocess tweet tokens; tokenize with BERT tokenizer; map tokens to IDs; pad token sequences; create attention masks
    '''
    input_ids = []
    atten_masks = []
    tag_groups = []
    for i, t in enumerate(tokens):
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        processed_tokens, processed_tags = preprocess_tweet(t, tags[i])
        encoded_dict = tokenizer.encode_plus(
            processed_tokens,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_TOKENIZED_TWEET_LENGTH,  # Pad & truncate all sentences.
            truncation=True,
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            is_split_into_words=True, # Already split into tokens
            return_tensors='pt', )  # Return pytorch tensors.

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        tag_features = np.zeros((MAX_TOKENIZED_TWEET_LENGTH,TAG_FEATURES))
        for i, group in enumerate(processed_tags):
            if i == MAX_TOKENIZED_TWEET_LENGTH: break
            for tag in group:
                tag_features[i][MAP_TAGS[tag]] += 1
        tag_groups.append(tag_features)

        # And its attention mask (simply differentiates padding from non-padding).
        atten_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(atten_masks, dim=0)

    tag_tensor = torch.tensor(tag_groups)

    # Convert label tokens to integers
    all_labels = set(labels)
    label_ints = np.array([LABEL_INDICES[label] for label in labels])
    label_tensor = torch.tensor(label_ints)

    return input_ids, attention_masks, label_tensor, tag_tensor


def get_dataloaders(input_ids, attention_masks, sentiment_labels, tag_sets, batch_size: int, train_share=DEFAULT_TRAIN_SHARE):
    # Split dataset into training and validation subsets
    dataset = TensorDataset(input_ids, attention_masks, sentiment_labels, tag_sets)
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
