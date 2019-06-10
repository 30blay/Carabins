from data_extraction.analysis import get_subject_metrics, get_traits_rapides_params, get_sigmalog_params
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np


X_variables = ['t0']
y_variable = 'post_exercise'

scoring = [
    'f1_weighted',
]

# load the dataset
df = pd.merge(get_subject_metrics(), get_traits_rapides_params(), left_on='id', right_on='subject_id')
df = df[X_variables + [y_variable]].dropna()
X = df[X_variables]
y = df[y_variable]

class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

models = [
    DummyClassifier(),
    DecisionTreeClassifier(),
]

print('Training on ' + str(X.shape[0]) + ' samples')

for model in models:
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=True)
    print(model.__class__)
    print('Train F1: ', cv_results['train_f1_weighted'].mean(), '    std: ', cv_results['train_f1_weighted'].std())
    print('Test F1: ', cv_results['test_f1_weighted'].mean(), '    std: ', cv_results['test_f1_weighted'].std())

