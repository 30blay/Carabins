from analysis import get_subject_metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = get_subject_metrics()
X = df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']]
y = df['avg_fatigue']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

svr = svm.SVR(kernel='linear')
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
print("Mean squared error: %.2f", mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f', r2_score(y_test, y_pred))

