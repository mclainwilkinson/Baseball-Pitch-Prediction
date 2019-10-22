import pandas as pd
import h5py
import numpy as np
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

    # transform columns to usable format
    df['score_diff'] = df['p_score'] - df['b_score']
    df['p_R'] = (df['p_throws'] == 'R') * 1.0
    df['p_L'] = (df['p_throws'] == 'L') * 1.0
    df['b_R'] = (df['stand'] == 'R') * 1.0
    df['b_L'] = (df['stand'] == 'L') * 1.0
    df['F'] = (df['simple_type'] == 'F') * 1.0
    df['C'] = (df['simple_type'] == 'C') * 1.0
    df['B'] = (df['simple_type'] == 'B') * 1.0

    return df

def convert(data_frame):
    # define columns for each vector
    init_cols = ['score_diff', 'inning', 'outs', 'p_R', 'p_L', 'b_R',
                'b_L', 'on_1b', 'on_2b', 'on_3b']
    seq_cols = ['F', 'C', 'B', 'b_count', 's_count']
    label_cols = ['F', 'C', 'B']

    # get ab_ids to loop through
    abs = data_frame['ab_id'].unique()

    # create lists for init, seq, and labels
    init = []
    seq = []
    labels = []

    # define seq and label shape to be of length 21
    seq_shape = (21, 5)
    label_shape = (21, 3)

    # for each ab, create vectors and add to list
    print('creating vectors.')
    abs_len = len(abs)
    for i, ab in enumerate(abs):
        if i % (abs_len // 100) == 0:
            print(i / abs_len * 100, '% done!')
        atbat = data_frame[data_frame['ab_id']==ab].reset_index()
        init_vec = atbat[init_cols].loc[0].to_numpy()
        seq_data = atbat[seq_cols].to_numpy()
        label_data = atbat[label_cols].to_numpy()

        seq_vecs = np.zeros(seq_shape)
        seq_vecs[:seq_data.shape[0],:5] = seq_data

        label_vecs = np.zeros(label_shape)
        label_vecs[:label_data.shape[0],:3] = label_data

        init.append(init_vec)
        seq.append(seq_vecs)
        labels.append(label_vecs)
    print('vectors created.')

    return np.array(init), np.array(seq), np.array(labels)

def store(inits, seqs, labels):
    # create h5 files for writing
    print('writing h5 files.')
    pitch_seq = 'pitches.h5'
    pitch_file = h5py.File(pitch_seq, 'w')

    init_vecs = 'inits.h5'
    init_file = h5py.File(init_vecs, 'w')

    label_mats = 'labels.h5'
    label_file = h5py.File(label_mats, 'w')

    # add data to h5 database
    pitch_file.create_dataset('pitch_seqs', data=seqs)
    init_file.create_dataset('init_vecs', data=inits)
    label_file.create_dataset('label_vecs', data=labels)

    # check data in h5 databases
    print(init_file.get('init_vecs')[0])
    print(pitch_file.get('pitch_seqs')[0])
    print(label_file.get('label_vecs')[0])

    # close files
    init_file.close()
    pitch_file.close()
    label_file.close()

def main(atbats, pitches):
    pitch_data = load_transform_data(atbats, pitches)
    init, seq, label  = convert(pitch_data)
    store(init, seq, label)
    print('h5 datasets created (pitches.h5, inits.h5, labels.h5).')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pitch_file', help='pitches csv file')
    parser.add_argument('-ab', '--atbat_file', help='atbats csv file')
    args = parser.parse_args()
    atbats = args.atbat_file
    pitches = args.pitch_file
    main(atbats, pitches)