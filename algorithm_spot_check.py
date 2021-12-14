import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

data_file = "dataset_timewarp1.csv"
# load dataset
dataset = pd.read_csv(data_file, header=None)
# split into inputs and outputs
values = dataset.values
X, y = values[:, :-1], values[:, -1]
# create a list of models to evaluate
models, names = list(), list()
# # logistic
# models.append(LogisticRegression())
# names.append('LR')
# knn
models.append(KNeighborsClassifier())
names.append('KNN')
# cart
models.append(DecisionTreeClassifier())
names.append('CART')
# svm
models.append(SVC())
names.append('SVM')
# random forest
models.append(RandomForestClassifier())
names.append('RF')
# gbm
models.append(GradientBoostingClassifier())
names.append('GBM')
# evaluate models
all_scores = list()
for i in range(len(models)):
    # create a pipeline for the model
    s = StandardScaler()
    p = Pipeline(steps=[('s', s), ('m', models[i])])
    scores = cross_val_score(p, X, y, scoring='accuracy', cv=5, n_jobs=-1)
    all_scores.append(scores)
    # summarize
    m, s = np.mean(scores)*100, np.std(scores)*100
    print('%s %.3f%% +/-%.3f' % (names[i], m, s))

# plot
pyplot.boxplot(all_scores, labels=names)
pyplot.title(data_file)
pyplot.xlabel("Models")
pyplot.ylabel("Accuracy")
pyplot.show()