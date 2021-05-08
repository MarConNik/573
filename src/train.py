import argparse
import random
import time

import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import load_train_data, BERT_MODEL_NAME, encode_strings, get_dataloaders
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification
from torch.utils.data import DataLoader


'''
E.g.:
python src/train.py --train-file data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll --model-file model.joblib
'''


# Default training hyperparameters
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_EPSILON = 1e-8
DEFAULT_NUM_EPOCHS = 4

DEFAULT_SEED = 634  # Generated pseudorandomly (out of 1000)
DEFAULT_BATCH_SIZE = 32


def log(message):
    print(message)


def train_model(
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_labels: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epsilon: float = DEFAULT_EPSILON,
    num_epochs: int = DEFAULT_NUM_EPOCHS
) -> BertPreTrainedModel:
    model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels; -, 0, + for sentiment
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    # Use CUDA if it's available
    # FIXME: Connor's PC has an AMD GPU, which doesn't use CUDA
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")

    # FIXME: The tutorial that we are following says to use the HuggingFace/Transformers version; there is a PyTorch
    #  version now, though.
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        eps=epsilon
    )

    training_steps = len(training_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # FIXME: Why run with warmup if we set warmup steps to 0?
        num_training_steps=training_steps
    )

    """ Training loop steps (from McCormick blog post):
        - Unpack our data inputs and labels
        - Load data onto the GPU for acceleration
        - Clear out the gradients calculated in the previous pass.
        - In pytorch the gradients accumulate by default (useful for things like RNNs) unless you explicitly clear them out.
        - Forward pass (feed input data through the network)
        - Backward pass (backpropagation)
        - Tell the network to update parameters with optimizer.step()
        - Track variables for monitoring progress
    
    Evaluation steps:
        - Unpack our data inputs and labels
        - Load data onto the GPU for acceleration
        - Forward pass (feed input data through the network)
        - Compute loss on our validation data and track variables for monitoring progress
    """

    # Set random seeds
    seed = DEFAULT_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # FIXME: Change from CUDA if you're using a different GPU architecture

    training_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Set model to train mode (this does NOT train the model)
        model.train()

        # Do once-through of training data per epoch
        # TODO: Track training and evaluation losses
        for step, batch in enumerate(training_dataloader):
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_sentiment_labels = batch[2].to(device)

            # Clear out previous gradients
            model.zero_grad()

            # Run model on batch (it accumulates loss gradient internally)
            result = model(
                batch_input_ids,
                attention_mask=batch_attention_masks,
                labels=batch_sentiment_labels,
                return_dict=True
            )
            batch_loss = result.loss
            batch_logits = result.logits

            log(f"Training loss of batch {step} of epoch {epoch}: {batch_loss.item()}")

            # Backpropagate the loss
            batch_loss.backward()

            # Clip the gradient norm to prevent "exploding gradient" problem
            # TODO: Get better understanding of this (why is model.parameters() an argument? Doesn't this work on the
            #  gradient, not the parameters?)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update model parameters based on gradient, learning rate, optimizer policy
            optimizer.step()

            # Update the learning rate based on step
            scheduler.step()

        training_end = time.time()
        epoch_train_minutes = (training_end - training_start) / 60
        log(f"Total train time of epoch {epoch}: {epoch_train_minutes}")

        # Do evaluation once per epoch
        # TODO: Finish evaluation part of training loop
        # Switch model to evaluation mode (store no gradients):
        model.eval()

        for step, batch in enumerate(validation_dataloader):
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_sentiment_labels = batch[2].to(device)

            # For some reason we have to globally disable gradient accumulation (in addition to setting model to `eval()`)
            with torch.no_grad():
                result = model(
                    batch_input_ids,
                    attention_mask=batch_attention_masks,
                    labels=batch_sentiment_labels,
                    return_dict=True
                )

            batch_loss = result.loss
            batch_logits = result.logits

            log(f"Validation loss of batch {step}, epoch {epoch}: {batch_loss.item()}")

    return model


if __name__ == '__main__':
    # Parser arguments:
    parser = argparse.ArgumentParser(description='classify records in a test set')
    parser.add_argument('--model-file', type=argparse.FileType('wb'),
                        help='path to save model to')
    parser.add_argument('--train-file',
                        type=argparse.FileType('r', encoding='latin-1'),
                        help='path to unlabelled testing data file')
    parser.add_argument('--dev-file',
                        type=argparse.FileType('r', encoding='latin-1'),
                        help='path to unlabelled testing data file')
    parser.add_argument('--batch-size', type=int,
                        default=DEFAULT_BATCH_SIZE,
                        help='training (and validation) batch size')
    args = parser.parse_args()

    # Get training tweets from file
    tweet_ids, tweets, sentiment_labels = load_train_data(args.train_file, bert=True)

    # Convert tweets to BERT-readable format
    input_ids, attention_masks, sentiment_labels = encode_strings(tweets, sentiment_labels)
    training_dataloader, validation_dataloader = get_dataloaders(
        input_ids=input_ids,
        attention_masks=attention_masks,
        sentiment_labels=sentiment_labels,
        batch_size=args.batch_size
    )

    # Count number of unique labels
    # FIXME: figure out a good way to make this no longer necessary
    num_labels = len(set(sentiment_labels))

    # Train model
    model = train_model(training_dataloader, validation_dataloader, num_labels)

    # Save model to joblib
    # joblib.dump((vectorizer, model), args.model_file)

    # Run model on test data
    # dev_ids, dev_tweets, dev_sentiments = load_train_data(args.dev_file)
    # dev_vectors = vectorizer.transform(dev_tweets)
    # dev_predictions = model.predict(dev_vectors)
