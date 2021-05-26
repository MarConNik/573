import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from .bert import BERT_MODEL_NAME


MODEL_FILENAME = 'model.pt'
LSTM_HIDDEN_SIZE = 100


class BertLSTMClassifier(nn.Module):
    def __init__(self, num_labels, bert_model=None, lstm_hidden_size=LSTM_HIDDEN_SIZE):
        """Initialize LSTM classifier given a BERT instance to use as a contextual embedding layer at the bottom
        """
        super(BertLSTMClassifier, self).__init__()

        # BERT Layer, for Contextual Embeddings
        if bert_model is None:
            bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.bert: BertModel = bert_model
        bert_embedding_size = bert_model.config.hidden_size

        # Bidirectional LSTM Layer
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=bert_embedding_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Final feed-forward linear layer for classification
        self.fc = nn.Linear(2*lstm_hidden_size, num_labels)

        # Cross entropy loss calculator
        # TODO: Incorporate class weights here
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # (Through the magic of tensors, this is all [foreach sequence])

        # Get contextual token embeddings from BERT
        contextual_embeddings = self.bert(input_ids, attention_mask)[0]

        # Feed that sequence of contextual token embeddings
        sequence_length: np.array = attention_mask.sum(dim=1).detach().cpu().numpy()
        packed_input = pack_padded_sequence(contextual_embeddings, sequence_length, batch_first=True, enforce_sorted=False)
        packed_lstm_output, _ = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True)

        # Get last hidden state of the forward (part of the) LSTM
        forward_outs = []
        for last, sequence_tensor in zip(sequence_length-1, lstm_output):
            last_hidden = sequence_tensor[min(last, len(sequence_tensor)-1), self.lstm_hidden_size:]
            forward_outs.append(last_hidden)
        forward_out = torch.stack(forward_outs)
        reverse_out = lstm_output[:, 0, :self.lstm_hidden_size]

        # Concatenate the two last LSTM hidden states, pass to feed-forward
        sequence_vector = torch.cat((forward_out, reverse_out), dim=-1)
        logits = self.fc(sequence_vector)

        if labels is None:
            return logits
        else:
            return self.loss(logits, labels)
