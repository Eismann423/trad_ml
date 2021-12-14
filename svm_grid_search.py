import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
# import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import os

file_number = 5
data_file = 'datasets_single\dataset_multiaug' + str(file_number) + '.csv'
output_file_name = "dataset_multiaug" + str(file_number)
output_file = r'C:\Users\Eisma\thesis\trad_ml\results\svm_aug_1000\\' + output_file_name + '_output.txt'
output_plot = r'C:\Users\Eisma\thesis\trad_ml\results\svm_aug_1000\\' + output_file_name + '_output.png'
# load dataset
dataset = pd.read_csv(data_file, header=None)
# split into inputs and outputs
values = dataset.values
X, y = values[:, :-1], values[:, -1]
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling the Train and Test feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the parameter grid based on the results of random search
params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
               {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# Performing CV to tune parameters for best SVM fit
svm_model = GridSearchCV(SVC(), params_grid, cv=5)
svm_model.fit(X_train_scaled, y_train)

final_model = svm_model.best_estimator_
y_pred = final_model.predict(X_test_scaled)

# Making the Confusion Matrix with testing data
disp = plot_confusion_matrix(final_model, X_test_scaled, y_test, display_labels=['V8-V9', 'V10-V11', 'V12-V13', 'V14-V15'],
                            cmap=plt.cm.Blues, normalize=None)

# Confusion Matrix made with training data
# disp = plot_confusion_matrix(final_model, X_train_scaled, y_train, display_labels=['V8-V9', 'V10-V11', 'V12-V13', 'V14-V15'],
#                             cmap=plt.cm.Blues, normalize=None)

disp.ax_.set_title("SVM from " + os.path.splitext(os.path.basename(data_file))[0] + " Training acc: " + str(round(final_model.score(X_train_scaled, y_train), 3)) +
                   " Testing acc: " + str(round(final_model.score(X_test_scaled, y_test), 3)))
plt.savefig(output_plot)


# Write model scores to file
f = open(output_file, "x")
# View the accuracy score
f.write('SVM grid search results for ' + data_file + '\n')
f.write('\n')
f.write('Best score for training data: %f\n' % svm_model.best_score_)

# View the best parameters for the model found using grid search
f.write('Best C: %d\n' % svm_model.best_estimator_.C)
f.write('Best Kernel: %s\n' % svm_model.best_estimator_.kernel)
f.write('Best Gamma: %s\n' % svm_model.best_estimator_.gamma)
f.write('\n')
f.write('Confusion Matrix:\n')
f.write(str(confusion_matrix(y_test, y_pred)))
f.write("\n\n")
f.write('Classification Report:\n')
f.write(classification_report(y_test, y_pred))
f.write('\n')
f.write("Training set score for SVM: %f\n" % final_model.score(X_train_scaled, y_train))
f.write("Testing set score for SVM: %f\n" % final_model.score(X_test_scaled, y_test))
f.close()


# # View the accuracy score
# print('Results for ' + data_file)
# print('Best score for training data:', svm_model.best_score_, "\n")
#
# # View the best parameters for the model found using grid search
# print('Best C:', svm_model.best_estimator_.C, "\n")
# print('Best Kernel:', svm_model.best_estimator_.kernel, "\n")
# print('Best Gamma:', svm_model.best_estimator_.gamma, "\n")
#
# final_model = svm_model.best_estimator_
# y_pred = final_model.predict(X_test_scaled)
#
# # Making the Confusion Matrix with testing data
# disp = plot_confusion_matrix(final_model, X_test_scaled, y_test, display_labels=['V8-V9', 'V10-V11', 'V12-V13', 'V14-V15'],
#                             cmap=plt.cm.Blues, normalize=None)
#
# # Confusion Matrix made with training data
# # disp = plot_confusion_matrix(final_model, X_train_scaled, y_train, display_labels=['V8-V9', 'V10-V11', 'V12-V13', 'V14-V15'],
# #                             cmap=plt.cm.Blues, normalize=None)
# disp.ax_.set_title("SVM from " + data_file + " Training acc: " + str(final_model.score(X_train_scaled, y_train)) +
#                    " Testing acc: " + str(final_model.score(X_test_scaled, y_test)))
# plt.show()
#
# print(confusion_matrix(y_test, y_pred))
# print("\n")
# print(classification_report(y_test, y_pred))
#
# print("Training set score for SVM: %f" % final_model.score(X_train_scaled, y_train))
# print("Testing set score for SVM: %f" % final_model.score(X_test_scaled, y_test))
#
# svm_model.score