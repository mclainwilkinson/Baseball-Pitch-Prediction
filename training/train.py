import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
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
data_file = 'baseballScaled.h5'
data = PitchDataset(data_file)
batch_size = 10
train_split = 0.7
train_loader, test_loader = split_dataset(data, batch_size, 1 - train_split, True, 31)
num_seqs = data.__len__() * train_split
print('training and testing datasets loaded. Batch size: %d train/test split: %.2f' % (batch_size, train_split))
print('training on %d pitch sequences.' % (num_seqs))

# define the model
model = PitchRNN(input_size=5, output_size=3, hidden_dim=8, n_layers=7, init_dim=10)
model.to(device)
print('model loaded')

# set training parameters and define loss
num_epochs = 25
lr = 0.0001
class0weight = 0.7
class1weight = 1.0
class2weight = 1.0
weights = torch.FloatTensor([class0weight, class1weight, class2weight])
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
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
print('beginning training')
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
    print('epoch', i + 1, ':', '%.3f' % l)

epoch_series = pd.Series([x+1 for x in range(num_epochs)])
loss_series = pd.Series(loss_list)
loss_frame = {'epoch': epoch_series, 'loss': loss_series}
loss_df = pd.DataFrame(loss_frame)
loss_df.to_csv('loss.csv', index=False)

torch.save(model.state_dict(), 'pitchpredWeighted.pkl')
print('model saved to pitchpredWeighted.pkl')

# TESTING
model.eval()
print('begin testing model')
num_test_seqs = data.__len__() * (1 - train_split)
print('testing on %d pitch sequences from test set' % (num_test_seqs))

# create confusion dict to store predictions
confusion = {0:[0, 0, 0], 1:[0, 0, 0], 2:[0, 0, 0]}

# get results
for init, pitch, seq_len, label in test_loader:
    inits = Variable(init).to(device)
    pitches = Variable(pitch).to(device)
    seq_lens = Variable(seq_len).to(device)
    labels = Variable(label).to(device)

    out, hidden = model(inits, pitches, seq_lens)
    outs, labs = format_outputs(out, labels, seq_lens)
    preds = torch.argmax(outs, dim=1)

    for p, l in zip(preds.cpu(), labs.cpu()):
        confusion[int(l)][int(p)] += 1
        # keys/cols are actual, vals/rows are pred
        # 0->F, 1->C, 2->B

# calculate stats
totals = [sum(confusion[p]) for p in confusion]
corrects = [confusion[p][p] for p in confusion]
overall_accuracy = sum(corrects) / sum(totals)
individual_accuracy = [c / t for c, t in zip(corrects, totals)]

# write results to pandas df and csv file
confusion = pd.DataFrame.from_dict(confusion)
confusion.columns = ['Fastball (actual)', 'Change Up (actual)', 'Breaking Ball (actual)']
confusion.index = ['Fastball (pred)', 'Change Up (pred)', 'Breaking Ball (pred)']
confusion['totals'] = confusion.sum(axis=1)
confusion.loc['totals'] = confusion.sum()
confusion.to_csv('confusionWeighted.csv', index=False)

# print results
print('the results are in...')
print('you\'re batting %.3f !!!!!' % (overall_accuracy))
print('individual accuracies:')
for l, a in zip(['fastball', 'change up', 'breaking ball'], individual_accuracy):
    print(l, '%.2f' % (a * 100.0), '%')
print('confusion matrix:')
print(confusion)
print('thanks for swinging!')
print('goodbye.')
