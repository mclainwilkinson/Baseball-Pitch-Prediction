import torch
import numpy as np
import h5py
from torch.utils.data.sampler import SubsetRandomSampler


class PitchDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(PitchDataset, self).__init__()
        h5_file = h5py.File(data)
        self.init_vecs = h5_file.get('init_vecs')
        self.pitch_seqs = h5_file.get('pitch_seqs')
        self.labels = h5_file.get('label_vecs')
        self.pitch_seq_lens = h5_file.get('seq_lens')

    def __getitem__(self, index):
        init_vec = torch.from_numpy(self.init_vecs[index]).float()
        pitch_seq = torch.from_numpy(self.pitch_seqs[index]).float()
        seq_len = torch.from_numpy(self.pitch_seq_lens[index]).float()
        label = torch.from_numpy(np.array(self.labels[index])).float()
        return init_vec, pitch_seq, seq_len, label

    def __len__(self):
        return self.labels.shape[0]

def split_dataset(dataset, batch, test_proportion, shuffle, r_seed):
    data_size = dataset.__len__()
    indices = list(range(data_size))
    split = int(np.floor(test_proportion * data_size))
    if shuffle:
        np.random.seed(r_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch,
                                                sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch,
                                                sampler=test_sampler)
    return train_loader, test_loader
