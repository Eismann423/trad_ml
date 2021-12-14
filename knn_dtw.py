from fastdtw import fastdtw
import numpy as np
from pandas import read_csv
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier"
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

file_number = 1
data_file = 'datasets_single\dataset_drift' + str(file_number) + '.csv'
output_file_name = "dataset_timewarp" + str(file_number)
output_file = r'C:\Users\Eisma\thesis\trad_ml\results\knn_aug_1000_dtw\\' + output_file_name  + '_output.txt'
output_plot = r'C:\Users\Eisma\thesis\trad_ml\results\knn_aug_1000_dtw\\' + output_file_name + '_output.png'

# load dataset
dataset = read_csv(data_file, header=None)
# split into inputs and outputs
values = dataset.values
X, y = values[:, :-1], values[:, -1]
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling the Train and Test feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # #toy dataset
# X = np.random.random((100,10))
# y = np.random.randint(0,2, (100))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#custom metric
def DTW(a, b):
    distance, path = fastdtw(a, b, dist=euclidean)
    return distance

#train
# parameters = {'n_neighbors':[1, 2, 4, 8]}
# clf = GridSearchCV(KNeighborsClassifier(metric=DTW), parameters, cv=3, verbose=1)
clf = KNeighborsClassifier(n_neighbors=1, metric=DTW)
# clf.fit(X_train, y_train)
clf.fit(X_train_scaled, y_train)



#evaluate"
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))