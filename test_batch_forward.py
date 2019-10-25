from model.net import PitchRNN
from model.data_loader import PitchDataset, split_dataset

data_file = 'baseball.h5'

print('creating dataset.')
data = PitchDataset(data_file)
print('dataset created.')

print('splitting dataset.')
train_loader, test_loader = split_dataset(data, 5, 0.3, True, 68)
print('dataset split.')

model = PitchRNN(input_size=5, output_size=3, hidden_dim=8, n_layers=3, init_dim=10)

for epoch, (init, pitch, seq_len, label) in enumerate(train_loader):
    if epoch > 1:
        break
    print('epoch:', epoch)
    out, hidden = model(init, pitch, seq_len)
    print(out)
