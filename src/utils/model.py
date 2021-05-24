import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel


LSTM_HIDDEN_SIZE = 100


class BertLSTMClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, lstm_hidden_size=LSTM_HIDDEN_SIZE):
        """Initialize LSTM classifier given a BERT instance to use as a contextual embedding layer at the bottom
        """
        super(BertLSTMClassifier, self).__init__()

        # BERT Layer, for Contextual Embeddings
        self.bert: BertModel = bert_model
        bert_embedding_size = bert_model.config.hidden_size

        # Bidirectional LSTM Layer
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm1 = nn.LSTM(
            input_size=bert_embedding_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
        )

        # Final feed-forward linear layer for classification
        self.fc = nn.Linear(2*lstm_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # (Through the magic of tensors, this is all [foreach sequence])

        # Get contextual token embeddings from BERT
        contextual_embeddings = self.bert(input_ids, attention_mask)

        # Feed that sequence of contextual token embeddings
        sequence_length = attention_mask.sum(dim=1)
        packed_input = pack_padded_sequence(contextual_embeddings, sequence_length)
        packed_lstm_output = self.lstm1(packed_input)
        lstm_output, sequence_length = pad_packed_sequence(packed_lstm_output, batch_first=True)

        # Get last hidden state of the forward (part of the) LSTM
        forward_out = lstm_output[:, sequence_length-1, :self.lstm_hidden_size]
        reverse_out = lstm_output[:, 0, self.lstm_hidden_size:]

        # Concatenate the two last LSTM hidden states, pass to feed-forward
        sequence_vector = torch.cat((forward_out, reverse_out), dim=1)
        return torch.softmax(self.fc(sequence_vector), 0)
