from torch import nn
from transformers import BertModel


LSTM_HIDDEN_SIZE = 100


class BertLSTMClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, lstm_hidden_size=LSTM_HIDDEN_SIZE):
        """Initialize LSTM classifier given a BERT instance to use as a contextual embedding layer at the bottom
        """
        # BERT Layer, for Contextual Embeddings
        self.bert: BertModel = bert_model
        bert_embedding_size = bert_model.config.hidden_size

        # Bidirectional LSTM Layer
        self.lstm1 = nn.LSTM(
            input_size=bert_embedding_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
        )

        # Final feed-forward linear layer for classification
        self.fc = nn.Linear(2*lstm_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # TODO: Finish implementing
        # (Through the magic of tensors, this is all [foreach sequence])

        # Get contextual token embeddings from BERT
        contextual_embeddings = self.bert(input_ids, attention_mask)

        # Feed that sequence of contextual token embeddings
        lstm_output = self.lstm1(contextual_embeddings)
