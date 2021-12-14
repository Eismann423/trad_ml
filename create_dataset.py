import numpy as np
import pandas as pd
import os

file_number = 5
dataset_name = 'dataset_multiaug' + str(file_number) + '.csv'


def load_dataset(prefix=''):
    # file paths for original data
    # path_forces = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass_svm\csv"
    # path_target = r"C:\Users\Eisma\thesis\nationals_data\svm_mappings\target.csv"
    # path_hand = r"C:\Users\Eisma\thesis\nationals_data\svm_mappings\hand.csv"

    # file paths for augmented data
    path_forces = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass_aug" \
                  r"\csv\multiaug" + str(file_number)
    path_target = r"C:\Users\Eisma\thesis\nationals_data\svm_mappings\target_aug_1000.csv"
    # path_hand = r"C:\Users\Eisma\thesis\nationals_data\svm_mappings\hand_aug_1000.csv"

    # load mapping files
    targets = pd.read_csv(path_target, header=0)
    # hand = pd.read_csv(path_hand, header=0)

    # load traces
    sequences = list();
    for name in os.listdir(path_forces):
        filename = path_forces + "\\" + name
        # df = pd.read_csv(filename, header=0, usecols=(1, 2, 3))
        df = pd.read_csv(filename, usecols=(0, 1, 2))
        values = df.values
        sequences.append(values)
    return sequences, targets.values[:,1]    # , hand.values[:,1] (don't need hand mappings for dataset creation)


# create a fixed 1d vector for each trace with output variable
def create_dataset(sequences, targets):
    # create the transformed dataset
    transformed = list()
    max_len = len(max(sequences, key=len))
    # process each trace in turn
    for i in range(len(sequences)):
        seq = sequences[i]
        vector = list()
        # pad with zeros
        new_seq = np.hstack([seq.transpose(), np.zeros([3, max_len-len(seq)])]).flatten()
        for j in range(len(new_seq)):
            vector.append(new_seq[j])
        # add output
        vector.append(targets[i])
        # store
        transformed.append(vector)
    # prepare array
    transformed = np.array(transformed)
    transformed = transformed.astype('float32')
    return transformed


# load dataset
sequences, targets = load_dataset()

# create dataset
dataset = create_dataset(sequences, targets)
print('Dataset: %s' % str(dataset.shape))
np.savetxt(dataset_name, dataset, delimiter=',')
