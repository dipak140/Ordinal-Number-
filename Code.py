from google.colab import files
uploaded = files.upload()
import io
df2 = pd.read_csv(io.BytesIO(uploaded['NumberData.csv']))
# Dataset is now stored in a Pandas Dataframe
feat = ['X1','X2', 'X3','X4', 'X5','X6','X7','X8','X9']
X = df2[feat]
label = ['Y']
y = df2[label]

X_train, X_test, y_train, y_test = X[:250], X[250:], y[:250], y[250:]

from sklearn import linear_model
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

clf.fit(X_train, y_train)

clf.predict(X_test.iloc[[5]])
y_test.iloc[[5]]
