import torch
from model.net import PitchRNN
from model.data_loader import PitchDataset, split_dataset

# get data from h5 dataset
data_file = 'baseball1.h5'
data = PitchDataset(data_file)
batch_size = 5
num_seqs = data.__len__()
train_loader, test_loader = split_dataset(data, batch_size, 0.3, True, 68)

# set training parameters and define loss
num_epochs = 5
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('using GPU')
else:
    device = torch.device("cpu")
    print('GPU not available, using CPU')

# define the model
model = PitchRNN(input_size=5, output_size=3, hidden_dim=8, n_layers=3, init_dim=10)
model.to(device)

# train the model
for epoch in range(num_epochs):
    for i, (init, pitch, seq_len, label) in enumerate(train_loader):
        inits = Variable(init).to(device)
        pitches = Variable(pitch).to(device)
        seq_lens = Variable(seq_len).to(device)
        labels = Variable(label).to(device)

        optimizer.zero_grad()
        out, hidden = model(inits, pitches, seq_lens)

        #<-------CALCULATE LOSS HERE----------->
        # loss = criterion(out, labels)
        # loss.backward()
        # optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, num_seqs // batch_size, loss.item()))

torch.save(model.state_dict(), 'pitchpred.pkl')

    
