import numpy as np
from pandas import read_csv
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# file_number = 1
# data_file = 'datasets_single\dataset_drift' + str(file_number) + '.csv'
# output_file_name = "dataset_timewarp" + str(file_number)
# output_file = r'C:\Users\Eisma\thesis\trad_ml\results\knn_aug_1000_dtw\\' + output_file_name  + '_output.txt'
# output_plot = r'C:\Users\Eisma\thesis\trad_ml\results\knn_aug_1000_dtw\\' + output_file_name + '_output.png'

# # load dataset
# dataset = read_csv(data_file, header=None)
# # split into inputs and outputs
# values = dataset.values
# X, y = values[:, :-1], values[:, -1]
# # split data into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#
# # Scaling the Train and Test feature set
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# #toy dataset
X = np.random.random((100,10))
y = np.random.randint(0,2, (100))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#custom metric
def DTW(a, b):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    return cumdist[an, bn]


def dtw_window(s, t):
    window = 3
    n, m = len(s), len(t)
    w = np.max([window, abs(n - m)])
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            dtw_matrix[i, j] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            cost = abs(s[i - 1] - t[j - 1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix

#train
# parameters = {'n_neighbors':[1, 2, 4, 8]}
# clf = GridSearchCV(KNeighborsClassifier(metric=DTW), parameters, cv=3, verbose=1)
clf = KNeighborsClassifier(n_neighbors=1, metric=dtw_window)
clf.fit(X_train, y_train)
# clf.fit(X_train_scaled, y_train)



#evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

