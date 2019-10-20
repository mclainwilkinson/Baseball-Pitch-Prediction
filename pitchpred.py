import torch
import torch.nn as nn
import h5py

# this file defines and trains the pitch prediction model
# source activate pytorch_p36

# load datasets
inits_dset = h5py.File('inits.h5')
pitches_dset = h5py.File('pitches.h5')
labels_dset = h5py.File('labels.h5')
inits = inits_dset.get('init_vecs')[0] # 10, 1
inits = torch.from_numpy(inits.T).unsqueeze(0).unsqueeze(0).float() # 10, 1, 1
pitches = pitches_dset.get('pitch_seqs')[0] # 21, 5
pitches = torch.from_numpy(pitches).unsqueeze(1).float() # 21, 1, 5
labels = labels_dset.get('label_vecs')[0] # 21, 3
labels = torch.from_numpy(labels).unsqueeze(1).float() # 21, 1, 3

# define model
class PitchRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, init_dim):
        super(PitchRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.init_dim = init_dim
        self.hidden_initializer1 = nn.Linear(self.init_dim, self.hidden_dim)
        self.hidden_initializer2 = nn.Linear(self.init_dim, self.hidden_dim)
        self.hidden_initializer3 = nn.Linear(self.init_dim, self.hidden_dim)
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, init_vec, seq_vec):
        h0_1 = self.hidden_initializer1(init_vec)
        h0_2 = self.hidden_initializer2(init_vec)
        h0_3 = self.hidden_initializer3(init_vec)
        h0 = torch.cat((h0_1, h0_2, h0_3), 0)
        out, hidden = self.rnn(seq_vec, h0)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden


# test forward pass
model = PitchRNN(input_size=5, output_size=3, hidden_dim=8, n_layers=3, init_dim=10) # h0 (n_layers, 1, hidden_dim)
output, h = model(inits, pitches)
print(output)
