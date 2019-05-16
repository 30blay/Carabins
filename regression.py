from analysis import get_subject_metrics
from sklearn.model_selection import train_test_split
from sklearn import svm, linear_model, neural_network
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

df = get_subject_metrics()
X = df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']]
y = df['avg_fatigue']
scoring=[
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

for model in models:
    cv_results = cross_validate(model, X, y, cv=3, scoring=scoring, return_train_score=True)
    print(model.__class__)
    print('Train R2: ', cv_results['train_r2'].mean(), '    std: ', cv_results['train_r2'].std())
    print('Test R2: ', cv_results['test_r2'].mean(), '    std: ', cv_results['test_r2'].std())
