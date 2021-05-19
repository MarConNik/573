import pytest
from utils import get_dataloaders, load_data, encode_strings


BATCH_SIZE = 32


def test_dataloaders():
    with open('data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll', 'r', encoding='latin-1') as training_file:
        tweet_ids, tweets, sentiment_labels = load_data(training_file)
    input_ids, attention_masks, sentiment_labels = encode_strings(tweets, sentiment_labels)
    training_dataloader, validation_dataloader = get_dataloaders(
        input_ids=input_ids,
        attention_masks=attention_masks,
        sentiment_labels=sentiment_labels,
        batch_size=BATCH_SIZE
    )

    for batch in training_dataloader:
        # Assert first batch is a full batch of correct length:
        input_ids, attention_masks, labels = batch
        assert input_ids.shape[0] == BATCH_SIZE
        break

    for batch in validation_dataloader:
        # Assert first batch is a full batch of correct length:
        input_ids, attention_masks, labels = batch
        assert attention_masks.shape[0] == BATCH_SIZE
        break
