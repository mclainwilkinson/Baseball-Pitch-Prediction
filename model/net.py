import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np


class PitchRNN(nn.Module):
    '''
    Pitch type (fastball, breaking ball, changeup) sequential classifier
    '''
    def __init__(self, input_size, output_size, hidden_dim, n_layers, init_dim):
        super(PitchRNN, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.init_dim = init_dim
        self.hidden_initializer1 = nn.Linear(self.init_dim, self.hidden_dim)
        self.hidden_initializer2 = nn.Linear(self.init_dim, self.hidden_dim)
        self.hidden_initializer3 = nn.Linear(self.init_dim, self.hidden_dim)
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, init_vecs, seq_vecs, seq_vec_lengths):
        batch_size = init_vecs.size(0)
        h0_1 = self.hidden_initializer1(init_vecs).unsqueeze(0)
        h0_2 = self.hidden_initializer2(init_vecs).unsqueeze(0)
        h0_3 = self.hidden_initializer3(init_vecs).unsqueeze(0)
        h0 = torch.cat((h0_1, h0_2, h0_3), 0)
        x_seqs = pack_padded_sequence(seq_vecs, seq_vec_lengths, batch_first=True, enforce_sorted=False)
        y_seqs, hidden = self.rnn(x_seqs, h0)
        y_seqs, _ = pad_packed_sequence(y_seqs, batch_first=True, total_length=22)
        out = self.fc(y_seqs)
        # do something to mask output here...
        return out, hidden
