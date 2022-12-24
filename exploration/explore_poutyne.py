import contextlib
import os
import pickle
import re
import sys
from io import TextIOBase

import fasttext
import fasttext.util
import requests
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence, PackedSequence
from torch.utils.data import DataLoader


from torchmetrics.classification import MulticlassAccuracy
from poutyne import set_seeds, Model
from poutyne.framework.metrics.predefined.accuracy import Accuracy
set_seeds(42)

cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

batch_size = 32
lr = 0.1

dimension = 2
num_layer = 1
bidirectional = False

lstm_network = nn.LSTM(input_size=dimension,
                       hidden_size=dimension,
                       num_layers=num_layer,
                       bidirectional=bidirectional,
                       batch_first=True)

input_dim = dimension # the output of the LSTM
tag_dimension = 3

fully_connected_network = nn.Linear(input_dim, tag_dimension)


types_dict = {"clicks":0, "carts":1, "orders":2}

train_data = pd.read_json('../raw-data/first_samples.json')
vectorized_data = []
for session_id in train_data.index:
    vectorized_data.append(([[dictonnary["aid"], dictonnary["ts"]] for dictonnary in train_data.loc[session_id]['events']],
                            [types_dict[dictonnary["type"]] for dictonnary in train_data.loc[session_id]['events']]))
vectorized_train_data = vectorized_data[:8000]
vectorized_validation_data = vectorized_data[8000:]
vectorized_test_data = vectorized_data

def pad_collate_fn(batch):
    """
    The collate_fn that can add padding to the sequences so all can have
    the same length as the longest one.

    Args:
        batch (List[List, List]): The batch data, where the first element
        of the tuple are the word idx and the second element are the target
        label.

    Returns:
        A tuple (x, y). The element x is a tensor of packed sequence .
        The element y is a tensor of padded tag indices. The word vectors are
        padded with vectors of 0s and the tag indices are padded with -100s.
        Padding with -100 is done because of the cross-entropy loss and the
        accuracy metric ignores the targets with values -100.
    """

    # This gets us two lists of tensors and a list of integer.
    # Each tensor in the first list is a sequence of word vectors.
    # Each tensor in the second list is a sequence of tag indices.
    # The list of integer consist of the lengths of the sequences in order.
    # for (seq_vectors, labels) in sorted(batch, key=lambda x: len(x[0]), reverse=True):
    #     print(seq_vectors)
    #     print(np.stack(seq_vectors))
    #     print(labels)
    #     print(torch.LongTensor(labels))
    #     print(torch.FloatTensor(np.stack(seq_vectors)))
    #     quit()
    sequences_vectors, sequences_labels, lengths = zip(*[
        (torch.FloatTensor(np.stack(seq_vectors)), torch.LongTensor(labels), len(seq_vectors))
        for (seq_vectors, labels) in sorted(batch, key=lambda x: len(x[0]), reverse=True)
    ])

    lengths = torch.LongTensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=0)
    pack_padded_sequences_vectors = pack_padded_sequence(
        padded_sequences_vectors, lengths.cpu(), batch_first=True
    )  # We pack the padded sequence to improve the computational speed during training

    padded_sequences_labels = pad_sequence(sequences_labels, batch_first=True, padding_value=-100)

    return pack_padded_sequences_vectors, padded_sequences_labels

train_loader = DataLoader(vectorized_train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
validation_loader = DataLoader(vectorized_validation_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
test_loader = DataLoader(vectorized_test_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

class FullNetWork(nn.Module):
    def __init__(self, lstm_network, fully_connected_network):
        super().__init__()
        self.hidden_state = None

        self.lstm_network = lstm_network
        self.fully_connected_network = fully_connected_network

    def forward(self, pack_padded_sequences_vectors: PackedSequence):
            """
                Defines the computation performed at every call.
            """
            lstm_out, self.hidden_state = self.lstm_network(pack_padded_sequences_vectors)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

            tag_space = self.fully_connected_network(lstm_out)
            return tag_space.transpose(-1, 1)  # We need to transpose since it's a sequence

full_network = FullNetWork(lstm_network, fully_connected_network)

print("Optimizing ...")
optimizer = optim.SGD(full_network.parameters(), lr)
loss_function = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.45, 0.45]))

model = Model(full_network, optimizer, loss_function,
              batch_metrics=[MulticlassAccuracy(num_classes=3, average="weighted")],
              device=device)

model.fit_generator(train_loader, validation_loader, epochs=10)

test_loss, test_acc = model.evaluate_generator(test_loader)

