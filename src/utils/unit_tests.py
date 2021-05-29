from .bert import encode_strings, get_dataloaders, MAX_TOKENIZED_TWEET_LENGTH, tokenizer
from .load import load_data
from .preprocess import preprocess_tweet


BATCH_SIZE = 32
SPANGLISH_TRAIN_PATH = 'data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll'
TEST_TWEETS_PATH = 'data/Semeval_2020_task9_data/test_tweets.txt'



def test_dataloaders():
    with open(SPANGLISH_TRAIN_PATH, 'r', encoding='latin-1') as training_file:
        tweet_ids, tweets, tags, sentiment_labels = load_data(training_file)
    input_ids, attention_masks, sentiment_labels, tag_groups = encode_strings(tweets, sentiment_labels, tags)
    training_dataloader, validation_dataloader = get_dataloaders(
        input_ids=input_ids,
        attention_masks=attention_masks,
        sentiment_labels=sentiment_labels,
        tag_sets=tag_groups,
        batch_size=BATCH_SIZE
    )

    for batch in training_dataloader:
        # Assert first batch is a full batch of correct length:
        input_ids, attention_masks, labels, tags = batch
        assert input_ids.shape[0] == BATCH_SIZE
        assert input_ids.shape[1] == MAX_TOKENIZED_TWEET_LENGTH
        assert len(input_ids.shape) == 2
        assert attention_masks.shape[0] == BATCH_SIZE
        assert attention_masks.shape[1] == MAX_TOKENIZED_TWEET_LENGTH
        break

    for batch in validation_dataloader:
        # Assert first batch is a full batch of correct length:
        input_ids, attention_masks, labels, tags = batch
        assert attention_masks.shape[0] == BATCH_SIZE
        break



def test_tokenizer():
    with open(TEST_TWEETS_PATH, 'r', encoding='latin-1') as training_file:
        tweet_ids, tweets, tags, sentiment_labels = load_data(training_file, with_labels=True)

    # assert tokenizer gets the same tokens as with the previous string input
    tweet_str = 'The best fall is... Fall in LOVE ❤ ️💌 Collar rojo $ 14.90 Pedidos 096.880.7384 #neckless #collar #accesorios http://t.co/6brIVHD2Xx'
    tokenized_tweet_str = tokenizer.encode_plus(tweet_str, add_special_tokens=True, max_length=MAX_TOKENIZED_TWEET_LENGTH, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt')
    tokenized_tweet = tokenizer.encode_plus(tweets[6], add_special_tokens=True, max_length=MAX_TOKENIZED_TWEET_LENGTH, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt', is_split_into_words=True)
    expected_tokens = ['[CLS]', 'The', 'best', 'fall', 'is', '.', '.', '.', 'Fall', 'in', 'LOVE', '[UNK]', '[UNK]', 'Coll', '##ar', 'rojo', '$', '14', '.', '90', 'Pe', '##dido', '##s', '096', '.', '880', '.', '738', '##4', '#', 'neck', '##less', '#', 'coll', '##ar', '#', 'acceso', '##rios', 'http', ':', '/', '/', 't', '.', 'co', '/', '6', '##br', '##IV']
    assert tokenized_tweet.tokens(0)[0:49] == expected_tokens
    assert tokenized_tweet_str.tokens(0)[0:49] == expected_tokens


def test_preprocessor_and_tags():
    with open(TEST_TWEETS_PATH, 'r', encoding='latin-1') as training_file:
        tweet_ids, tweets, tags, sentiment_labels = load_data(training_file, with_labels=True)

    # assert tokenizer batch object has same values encode_strings output
    preprocessed_tweet, preprocessed_tag = preprocess_tweet(tweets[4], tags[4])
    tokenized_tweet = tokenizer.encode_plus(preprocessed_tweet, add_special_tokens=True, max_length=MAX_TOKENIZED_TWEET_LENGTH, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt', is_split_into_words=True)
    expected_processed_tokens = ['@USER', '@USER', '@USER', '@USER', '@USER', 'Modi', 'ne', 'yeh', 'sab', 'nhi', 'karwaya', 'hai', '.', 'Yeh', 'yagan', 'k', 'politi', '…', 'HTTPURL']
    assert preprocessed_tweet == expected_processed_tokens
    token_ids, _, _, tag_tensor = encode_strings(tweets, sentiment_labels, tags)
    assert list(tokenized_tweet.input_ids[0].numpy()) == list(token_ids[4].numpy())
    token_indices = tokenized_tweet.word_ids(0)
    # user mentions are split into three BERT tokens and maintain connection with original token index
    expected_split = ['@', 'US', '##ER']
    assert tokenized_tweet.tokens(0)[1:4] == expected_split
    assert tokenized_tweet.tokens(0)[4:7] == expected_split
    assert token_indices[1:4] == [0, 0, 0]
    assert token_indices[4:7] == [1, 1, 1]
    # check that the tag counts per token are encoded in a tensor grouped with the BERT tokens
    assert preprocessed_tag[0] == ['O', 'Eng']
    assert preprocessed_tag[1] == ['O', 'Eng', 'O', 'Eng']
    tag_feature_0 = [1, 0, 0, 0, 0, 1, 0, 0, 0]
    tag_feature_1 = [2, 0, 0, 0, 0, 2, 0, 0, 0]
    tag_features_per_token = tag_tensor[4][0:7].numpy().astype(int)
    assert tag_feature_0 == list(tag_features_per_token[1]) == list(tag_features_per_token[2]) == list(tag_features_per_token[3])
    assert tag_feature_1 == list(tag_features_per_token[4]) == list(tag_features_per_token[5]) == list(tag_features_per_token[6])
