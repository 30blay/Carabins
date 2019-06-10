from data_extraction.analysis import get_subject_metrics, get_traits_rapides_params, get_sigmalog_params
from sklearn import svm, linear_model, neural_network
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from data_extraction.data_extraction import medical_traits, delta_log_params
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


# define regression parameters
X_variables = ['t0']
y_variable = 'avg_fatigue'

scoring = [
    'r2',
    'neg_mean_absolute_error',
]

# add calculated columns
df = pd.merge(get_subject_metrics(), get_traits_rapides_params(), left_on='id', right_on='subject_id')
df['scap_diff'] = (df['scap_d1'] + df['scap_d2'])/2 - (df['scap_g1'] + df['scap_g2'])/2
df['flex_diff'] = (df['flex_d1'] + df['flex_d2'])/2 - (df['flex_g1'] + df['flex_g2'])/2
df['re_gh_diff'] = (df['re_gh_d1'] + df['re_gh_d2'])/2 - (df['re_gh_g1'] + df['re_gh_g2'])/2

# transform data
df = df[X_variables + [y_variable]].dropna()
X = df[X_variables]
y = df[y_variable]

X_embedded = TSNE(n_components=1).fit_transform(X)
plt.scatter(X_embedded, y)
plt.show()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# train and test
models = [
    DummyRegressor('mean'),
    linear_model.Ridge(),
    linear_model.Lasso(),
    linear_model.SGDRegressor(),
    svm.SVR(gamma='auto'),
    neural_network.MLPRegressor((4,), max_iter=20000)
]

print('Training on ' + str(X.shape[0]) + ' samples')

for model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=scoring, return_train_score=True)
    print(model.__class__)
    print('Train R2: ', cv_results['train_r2'].mean(), '    std: ', cv_results['train_r2'].std())
    print('Test R2: ', cv_results['test_r2'].mean(), '    std: ', cv_results['test_r2'].std())
