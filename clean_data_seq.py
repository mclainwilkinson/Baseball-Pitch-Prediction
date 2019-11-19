import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse

'''
This script loads and transforms the data in 3 csv files (atbats, pitches,
and player_names), transforming it into sequences of vectors suitable
for the deep learning pitch type prediction. The final sequences and vectors
for each atbat are stored in an h5 file for easly loading and retrieval.
'''

def load_transform_data(abs, pitches):
    #read in csv files
    print('reading in data.')
    atbats = pd.read_csv(abs)
    pitches = pd.read_csv(pitches)

    # change key data type
    print('transforming data.')
    pitches['ab_id'] = pitches['ab_id'].astype(int)

    # get columns of interest
    ab_cols = ['inning', 'p_score', 'p_throws', 'stand', 'top', 'ab_id']
    pitch_cols = ['pitch_type', 'b_score', 'b_count', 's_count', 'outs',
                 'on_1b', 'on_2b', 'on_3b', 'ab_id', 'type']

    # merge pitches and atbats into 1 df
    df = pitches.merge(atbats, left_on='ab_id', right_on='ab_id')

    # drop abs containing infrequent pitch types
    low_counts = ['AB', 'FA', 'UN', 'PO', 'FO', 'EP', 'KN', 'IN', 'SC']
    dropabs = df[df['pitch_type'].isin(low_counts)]['ab_id'].unique().tolist()
    df = df[~df['ab_id'].isin(dropabs)]

    # convert pitch types to generalized pitch types
    pitch_conversions = {
        'CH': 'C',
        'CU': 'B',
        'KC': 'B',
        'SL': 'B',
        'FC': 'F',
        'FF': 'F',
        'FS': 'F',
        'FT': 'F',
        'SI': 'F'
    }

    df = df.dropna()
    df['simple_type'] = df['pitch_type'].apply(lambda x: pitch_conversions[x])

    pitch_class = {
        'F': 0,
        'C': 1,
        'B': 2
    }

    # transform columns to usable format and scale values
    df['score_diff'] = df['p_score'] - df['b_score']
    score_scaler = StandardScaler()
    df['score_diff'] = score_scaler.fit_transform(df['score_diff'].values.reshape(-1, 1))
    inning_scaler = MinMaxScaler()
    df['inning'] = inning_scaler.fit_transform(df['inning'].values.reshape(-1, 1))
    df['p_R'] = (df['p_throws'] == 'R') * 1.0
    df['p_L'] = (df['p_throws'] == 'L') * 1.0
    df['b_R'] = (df['stand'] == 'R') * 1.0
    df['b_L'] = (df['stand'] == 'L') * 1.0
    df['pitch_class'] = df['simple_type'].apply(lambda x: pitch_class[x])
    df['F'] = (df['simple_type'] == 'F') * 1.0
    df['C'] = (df['simple_type'] == 'C') * 1.0
    df['B'] = (df['simple_type'] == 'B') * 1.0

    return df

def convert(data_frame):
    # define columns for each vector
    seq_cols = ['F', 'C', 'B', 'b_count', 's_count', 'score_diff',
               'inning', 'outs', 'p_R', 'p_L', 'b_R', 'b_L', 'on_1b',
               'on_2b', 'on_3b']
    label_cols = 'pitch_class'

    # get ab_ids to loop through
    abs = data_frame['ab_id'].unique()

    # create lists for seqs, labels, lens
    seq = []
    labels = []
    lens = []

    # define seq and label shape to be of length 21
    seq_shape = (21, 15)
    label_shape = (22,)

    # for each ab, create vectors and add to list
    print('creating vectors.')
    abs_len = len(abs)
    for i, ab in enumerate(abs):
        if i % (abs_len // 100) == 0:
            print('%.2f percent done!' % (i / abs_len * 100))
        atbat = data_frame[data_frame['ab_id']==ab].reset_index()
        seq_data = atbat[seq_cols].to_numpy()
        label_data = atbat[label_cols].to_numpy()

        seq_len = seq_data.shape[0]
        seq_vecs = np.zeros(seq_shape)
        seq_vecs[:seq_data.shape[0]-1,:15] = seq_data[:-1,:]
        seq_vecs = np.insert(seq_vecs, 0, 0, axis=0)

        label_vecs = np.full(label_shape, np.nan)
        label_vecs[:len(label_data)] = label_data

        seq.append(seq_vecs)
        labels.append(label_vecs)
        lens.append(seq_len)
    print('vectors created.')

    return np.array(seq), np.array(labels), np.array(lens)

def store(seqs, labels, lens):
    # create h5 files for writing
    print('writing h5 file.')
    h5_file = 'baseballSeq.h5'
    bballDB = h5py.File(h5_file, 'w')

    # add data to h5 database
    bballDB.create_dataset('pitch_seqs', data=seqs)
    bballDB.create_dataset('label_vecs', data=labels)
    bballDB.create_dataset('seq_lens', data=lens)

    # check data in h5 database
    print(bballDB.get('pitch_seqs')[0])
    print(bballDB.get('label_vecs')[0])
    print(bballDB.get('seq_lens')[0])

    # close file
    bballDB.close()

def main(atbats, pitches):
    pitch_data = load_transform_data(atbats, pitches)
    seq, label, length = convert(pitch_data)
    store(seq, label, length)
    print('h5 dataset created (baseballSeq.h5)')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pitch_file', help='pitches csv file')
    parser.add_argument('-ab', '--atbat_file', help='atbats csv file')
    args = parser.parse_args()
    atbats = args.atbat_file
    pitches = args.pitch_file
    main(atbats, pitches)
