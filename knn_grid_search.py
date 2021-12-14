from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from scipy.spatial import distance
import os

# file_number = 4
# data_file = 'datasets_single\dataset_timewarp' + str(file_number) + '.csv'
# output_file_name = "dataset_timewarp" + str(file_number)
# output_file = r'C:\Users\Eisma\thesis\trad_ml\results\knn_aug_1000\\' + output_file_name  + '_output.txt'
# output_plot = r'C:\Users\Eisma\thesis\trad_ml\results\knn_aug_1000\\' + output_file_name + '_output.png'

def train_knn(data_file, output_file, output_plot):
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

    # Grid search using sklearn GridSearchCV
    grid_params = {'n_neighbors': [3, 5, 11, 19],
                   'weights': ['uniform', 'distance'],
                   'metric': ['euclidean', 'manhattan']}

    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=-1)
    gs.fit(X_train_scaled, y_train)

    final_model = gs.best_estimator_
    y_pred = final_model.predict(X_test_scaled)

    # Create confusion matrix plot
    disp = plot_confusion_matrix(final_model, X_test_scaled, y_test,
                                 display_labels=['V8-V9', 'V10-V11', 'V12-V13', 'V14-V15'],
                                 cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title("KNN from " + os.path.splitext(os.path.basename(data_file))[0] + " Training acc: " + str(round(final_model.score(X_train_scaled, y_train), 3)) +
                       " Testing acc: " + str(round(final_model.score(X_test_scaled, y_test), 3)))
    plt.savefig(output_plot)

    # Write model scores to file
    f = open(output_file, "x")
    # View the accuracy score
    f.write('KNN grid search results for ' + data_file + '\n')
    f.write('\n')
    f.write('Best score for training data: %f\n' % gs.best_score_)

    # View the best parameters for the model found using grid search
    f.write('Best K-value: %d\n' % gs.best_estimator_.n_neighbors)
    f.write('Best Weight: %s\n' % gs.best_estimator_.weights)
    f.write('Best Metric: %s\n' % gs.best_estimator_.metric)
    f.write('\n')
    f.write('Confusion Matrix:\n')
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write("\n\n")
    f.write('Classification Report:\n')
    f.write(classification_report(y_test, y_pred))
    f.write('\n')
    f.write("Training set score for KNN: %f\n" % final_model.score(X_train_scaled, y_train))
    f.write("Testing set score for KNN: %f\n" % final_model.score(X_test_scaled, y_test))
    f.close()
    return

for subdir, dirs, files in os.walk('datasets_single'):
    for file in files:
        if "multiaug" in file:
            output_name = os.path.splitext(file)[0]
            input_file = 'datasets_single\\' + file
            output_file = r'C:\Users\Eisma\thesis\trad_ml\results\knn_aug_1000\\' + output_name + '_output.txt'
            output_plot = r'C:\Users\Eisma\thesis\trad_ml\results\knn_aug_1000\\' + output_name + '_output.png'

            train_knn(input_file, output_file, output_plot)

            print("----------Finished with " + output_name + "----------")


