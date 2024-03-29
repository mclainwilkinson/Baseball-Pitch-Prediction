import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np


class PitchRNN(nn.Module):
    '''
    Pitch type (fastball, breaking ball, changeup) sequential classifier
    '''
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(PitchRNN, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, seq_vecs, seq_vec_lengths):
        batch_size = seq_vec_lengths.size(0)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        if torch.cuda.is_available():
            h0.cuda()
        x_seqs = pack_padded_sequence(seq_vecs, seq_vec_lengths, batch_first=True, enforce_sorted=False)
        y_seqs, hidden = self.rnn(x_seqs, h0)
        y_seqs, _ = pad_packed_sequence(y_seqs, batch_first=True, total_length=22)
        out = self.fc(y_seqs)
        return out, hidden
