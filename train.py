import torch
import torch.nn as nn
from torch.autograd import Variable
from model.net import PitchRNN
from model.data_loader import PitchDataset, split_dataset


# check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('using GPU')
else:
    device = torch.device("cpu")
    print('GPU not available, using CPU')

# get data from h5 dataset
data_file = 'baseball1.h5'
data = PitchDataset(data_file)
batch_size = 50
num_seqs = data.__len__()
train_loader, test_loader = split_dataset(data, batch_size, 0.3, True, 68)
print('training and testing datasets loaded')

# define the model
model = PitchRNN(input_size=5, output_size=3, hidden_dim=8, n_layers=3, init_dim=10)
model.to(device)

# set training parameters and define loss
num_epochs = 5
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# calculate loss
def format_outputs(output, label, seq_len):
    os = []
    ls = []
    for o, l, s in zip(output, label, seq_len):
        os.append(o[:s.item(),:])
        ls.append(l[:s.item()])
    return torch.cat(os, 0), torch.cat(ls, 0).long()

# keep track of loss
loss_list = []

# train the model
for epoch in range(num_epochs):
    avg_loss = []
    for i, (init, pitch, seq_len, label) in enumerate(train_loader):
        inits = Variable(init).to(device)
        pitches = Variable(pitch).to(device)
        seq_lens = Variable(seq_len).to(device)
        labels = Variable(label).to(device)

        optimizer.zero_grad()
        out, hidden = model(inits, pitches, seq_lens)

        outs, labs = format_outputs(out, labels, seq_lens)
        loss = criterion(outs, labs)
        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, num_seqs // batch_size, loss.item()))

    loss_list.append(np.mean(avg_loss))

print('training complete. Progression of loss is as follows:')
for i, l in enumerate(loss_list):
    print('epoch', i, ':', l)

torch.save(model.state_dict(), 'pitchpred.pkl')
