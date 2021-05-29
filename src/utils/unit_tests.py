from .bert import encode_strings, get_dataloaders, MAX_TOKENIZED_TWEET_LENGTH
from .load import load_data
from .preprocess import preprocess_tweet


BATCH_SIZE = 32
SPANGLISH_TRAIN_PATH = 'data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll'


def test_dataloaders():
    with open(SPANGLISH_TRAIN_PATH, 'r', encoding='latin-1') as training_file:
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
        assert input_ids.shape[1] == MAX_TOKENIZED_TWEET_LENGTH
        assert len(input_ids.shape) == 2
        assert attention_masks.shape[0] == BATCH_SIZE
        assert attention_masks.shape[1] == MAX_TOKENIZED_TWEET_LENGTH
        break

    for batch in validation_dataloader:
        # Assert first batch is a full batch of correct length:
        input_ids, attention_masks, labels = batch
        assert attention_masks.shape[0] == BATCH_SIZE
        break


def test_fix_encoding():
    with open(SPANGLISH_TRAIN_PATH, 'r', encoding='latin-1') as training_file:
        tweet_ids, tweets, sentiment_labels = load_data(training_file, with_labels=True)

    good_string = 'The best fall is... Fall in LOVE ❤ ️💌 Collar rojo $ 14.90 Pedidos 096.880.7384 #neckless #collar #accesorios http://t.co/6brIVHD2Xx'
    assert tweets[11999] == good_string


def test_preprocessing():
    tweet = 'The best fall is... Fall in LOVE ❤ ️💌 Collar rojo $ 14.90 Pedidos 096.880.7384 #neckless #collar #accesorios http://t.co/6brIVHD2Xx'
    preprocessed_tweet = preprocess_tweet(tweet)
    assert preprocessed_tweet == 'The best fall is... Fall in LOVE red heart  love letter Collar rojo $ 14.90 Pedidos 096.880.7384 #neckless #collar #accesorios HTTPURL'
