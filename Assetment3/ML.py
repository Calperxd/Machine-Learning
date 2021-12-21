import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('data.csv')
df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df['z'] = pd.to_numeric(df['z'], errors='coerce')

x = df.iloc[:, 1:4]
y = df.iloc[:, 4:5]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)


#Min-Max
mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.fit_transform(x_test)

#sc
sc = StandardScaler(with_mean=False)
sc.fit_transform(x_train)
sc.fit_transform(x_test)


x_train = pd.DataFrame(x_train, columns = ['x','y','z'])
x_test = pd.DataFrame(x_test, columns = ['x','y','z'])

## average value
x_train['avgX'] = x_train['x'].mean()
x_train['avgY'] = x_train['y'].mean()
x_train['avgZ'] = x_train['z'].mean()

x_test['avgX'] = x_test['x'].mean()
x_test['avgY'] = x_test['y'].mean()
x_test['avgZ'] = x_test['z'].mean()


# Minimum value
x_train['minX'] = x_train['x'].min()
x_train['minY'] = x_train['y'].min()
x_train['minZ'] = x_train['z'].min()

x_test['minX'] = x_test['x'].min()
x_test['minY'] = x_test['y'].min()
x_test['minZ'] = x_test['z'].min()

# Maximum value
x_train['maxX'] = x_train['x'].max()
x_train['maxY'] = x_train['y'].max()
x_train['maxZ'] = x_train['z'].max()

x_test['maxX'] = x_test['x'].max()
x_test['maxY'] = x_test['y'].max()
x_test['maxZ'] = x_test['z'].max()


# desvio
x_train['metricX'] = 0
x_train['metricY'] = 0
x_train['metricZ'] = 0

#Métrica de teste
for i in range(len(x_train)):
    x_train.loc[i, 'metricX'] = np.sqrt(np.abs(x_train.loc[i, 'avgX'] - x_train.loc[i, 'x']))
    x_train.loc[i, 'metricY'] = np.sqrt(np.abs(x_train.loc[i, 'avgY'] - x_train.loc[i, 'y']))
    x_train.loc[i, 'metricZ'] = np.sqrt(np.abs(x_train.loc[i, 'avgZ'] - x_train.loc[i, 'z']))

for i in range(len(x_test)):
    x_test.loc[i,'metricX'] = np.sqrt(np.abs(x_test.loc[i, 'avgX'] - x_test.loc[i, 'x']))
    x_test.loc[i, 'metricY'] = np.sqrt(np.abs(x_test.loc[i, 'avgY'] - x_test.loc[i, 'y']))
    x_test.loc[i, 'metricZ'] = np.sqrt(np.abs(x_test.loc[i, 'avgZ'] - x_test.loc[i, 'z']))

# desvio
x_train['stdX'] = 0
x_train['stdY'] = 0
x_train['stdZ'] = 0

for i in range(len(x_train)):
    x_train.loc[i,'stdX'] = x_train.loc[i, 'avgX'] - x_train.loc[i, 'x']
    x_train.loc[i, 'stdY'] = x_train.loc[i, 'avgY'] - x_train.loc[i, 'y']
    x_train.loc[i, 'stdZ'] = x_train.loc[i, 'avgZ'] - x_train.loc[i, 'z']

for i in range(len(x_test)):
    x_test.loc[i,'stdX'] = x_test.loc[i, 'avgX'] - x_test.loc[i, 'x']
    x_test.loc[i, 'stdY'] = x_test.loc[i, 'avgY'] - x_test.loc[i, 'y']
    x_test.loc[i, 'stdZ'] = x_test.loc[i, 'avgZ'] - x_test.loc[i, 'z']


x_train = x_train.fillna(x_train.mean())
y_train = y_train.fillna(x_train.mean())
x_test = x_test.fillna(0)
y_test = y_test.fillna(0)


x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

# Training knn
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print('knn: ' + str(knn.score(x_train, np.ravel(y_train, order='C'))))
print("knn accuracy:", metrics.accuracy_score(np.ravel(y_test, order='C'), y_pred))

#logistic regression
lgr = LogisticRegression(solver='newton-cg', penalty='none', C= 30)
lgr.fit(x_train, y_train)
y_pred = lgr.predict(x_test)
print('lgr: ' + str(lgr.score(x_train, np.ravel(y_train, order='C'))))
print("lgr accuracy:",metrics.accuracy_score(np.ravel(y_test, order='C'), y_pred))

#SVC
svc = SVC(probability=True, random_state=1)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print('svc: ' + str(svc.score(x_train, np.ravel(y_train, order='C'))))
print("svc accuracy:",metrics.accuracy_score(np.ravel(y_test, order='C'), y_pred))

#RandomFlorest
rdf = RandomForestClassifier(n_estimators=230, min_samples_split=5, min_samples_leaf=1, max_features='auto',max_depth=80, n_jobs=-1)
rdf.fit(x_train, y_train)
y_pred = rdf.predict(x_test)
print('rdf: ' + str(rdf.score(x_train, np.ravel(y_train, order='C'))))
print("rdf accuracy:",metrics.accuracy_score(np.ravel(y_test, order='C'), y_pred))

#LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
y_pred = lda.predict(x_test)
print('lda: ' + str(lda.score(x_train, np.ravel(y_train, order='C'))))
print("lda accuracy:",metrics.accuracy_score(np.ravel(y_test, order='C'), y_pred))



