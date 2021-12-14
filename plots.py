import pandas as pd
import os
import numpy as np
from matplotlib import pyplot

def load_dataset(prefix=''):
    # file paths for original data
    # path_forces = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass_svm\csv"
    # path_target = r"C:\Users\Eisma\thesis\nationals_data\svm_mappings\target.csv"
    # path_hand = r"C:\Users\Eisma\thesis\nationals_data\svm_mappings\hand.csv"

    # file paths for augmented data
    path_forces = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass_noise\csv\noise2"
    path_target = r"C:\Users\Eisma\thesis\nationals_data\svm_mappings\target_aug.csv"
    path_hand = r"C:\Users\Eisma\thesis\nationals_data\svm_mappings\hand_aug.csv"

    # load mapping files
    targets = pd.read_csv(path_target, header=0)
    hand = pd.read_csv(path_hand, header=0)

    # load traces
    sequences = list();
    for name in os.listdir(path_forces):
        filename = path_forces + "\\" + name
        # df = pd.read_csv(filename, header=0, usecols=(1, 2, 3))
        df = pd.read_csv(filename, usecols=(0, 1, 2))
        values = df.values
        sequences.append(values)
    return sequences, targets.values[:,1], hand.values[:,1]

#fit a linear regression function and return the predicted values for the series
def regress(y):
    # define input as the time step
    X = np.array([i for i in range(len(y))]).reshape(len(y), 1)
    # fit lenear regression via least squares
    b = np.linalg.lstsq(X, y)[0][0]
    # predict trend on time step
    yhat = b * X[:,0]
    return yhat

# load dataset
sequences, targets, hand = load_dataset()

# summarize class breakdown
class1 = len(targets[targets==1])
class2 = len(targets[targets==2])
class3 = len(targets[targets==3])
class4 = len(targets[targets==4])
print('Class=1: %d %.3f%%' % (class1, class1/len(targets)*100))
print('Class=2: %d %.3f%%' % (class2, class2/len(targets)*100))
print('Class=3: %d %.3f%%' % (class3, class3/len(targets)*100))
print('Class=4: %d %.3f%%' % (class4, class4/len(targets)*100))

# histogram for each anchor point
all_rows = np.vstack(sequences)
pyplot.figure()
variables = [0, 1, 2]
for v in variables:
    pyplot.subplot(len(variables), 1, v+1)
    pyplot.hist(all_rows[:, v], bins=20)
pyplot.show()

# histogram for trace lengths
trace_lengths = [len(x) for x in sequences]
pyplot.hist(trace_lengths, bins = 50)
pyplot.show()

# group sequences by hand
hands = [1,2]
seq_hand = dict()
for hand in hands:
    seq_hand[hand] = [sequences[j] for j in range(len(hands)) if hands[j]==hand]

# plot one example of a trace for each path
pyplot.figure()
for i in hands:
    pyplot.subplot(len(hands), 1, i)
    # line plot each variable
    for j in [0, 1, 2]:
        pyplot.plot(seq_hand[i][0][:, j], label='Anchor ' + str(j+1))
    pyplot.title('Hand ' + str(i), y=0, loc='left')
pyplot.show()

# plot series for a single trace with trend
seq = sequences[0]
variables = [0, 1, 2]
pyplot.figure()
for i in variables:
    pyplot.subplot(len(variables), 1, i+1)
    # plot the series
    pyplot.plot(seq[:,i])
    # plot the trend
    pyplot.plot(regress(seq[:,i]))
pyplot.show()