from data_extraction.analysis import get_subject_metrics
from sklearn import svm, linear_model, neural_network
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from data_extraction.data_extraction import medical_traits


X_variables = medical_traits
y_variable = 't0'

df = get_subject_metrics()[X_variables + [y_variable]].dropna()
X = df[X_variables]
y = df[y_variable]
scoring = [
    'r2',
    'neg_mean_absolute_error',
]


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

models = [
    linear_model.Ridge(),
    linear_model.Lasso(),
    linear_model.SGDRegressor(),
    svm.SVR(gamma='auto'),
    neural_network.MLPRegressor((4,), max_iter=20000)
]

print('Training on ' + str(X.shape[0]) + ' samples')

for model in models:
    cv_results = cross_validate(model, X, y, cv=3, scoring=scoring, return_train_score=True)
    print(model.__class__)
    print('Train R2: ', cv_results['train_r2'].mean(), '    std: ', cv_results['train_r2'].std())
    print('Test R2: ', cv_results['test_r2'].mean(), '    std: ', cv_results['test_r2'].std())
