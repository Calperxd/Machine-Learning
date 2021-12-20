import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

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

x_train = pd.DataFrame(x_train, columns = ['x','y','z'])
x_test = pd.DataFrame(x_test, columns = ['x','y','z'])

## average value
x_train['avgX'] = x_train['x'].mean()
x_train['avgY'] = x_train['y'].mean()
x_train['avgZ'] = x_train['z'].mean()

# Minimum value
x_train['minX'] = x_train['x'].min()
x_train['minY'] = x_train['y'].min()
x_train['minZ'] = x_train['z'].min()

# Maximum value
x_train['maxX'] = x_train['x'].max()
x_train['maxY'] = x_train['y'].max()
x_train['maxZ'] = x_train['z'].max()

# desvio value
x_train['stdX'] = 0
x_train['stdY'] = 0
x_train['stdZ'] = 0

for i in range(len(x_train)):
    x_train.loc[i,'stdX'] = x_train.loc[i, 'avgX'] - x_train.loc[i, 'x']
    x_train.loc[i, 'stdY'] = x_train.loc[i, 'avgY'] - x_train.loc[i, 'y']
    x_train.loc[i, 'stdZ'] = x_train.loc[i, 'avgZ'] - x_train.loc[i, 'z']


x_train = x_train.fillna(x_train.mean())
y_train = y_train.fillna(x_train.mean())
x_test = x_test.fillna(x_test.mean())
y_test = y_test.fillna(y_test.mean())


x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

# Training
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)


# Calculate the accuracy of the model
print(knn.score(x_train, np.ravel(y_train)))

#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]

#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

#Use GridSearch
clf = GridSearchCV(knn, hyperparameters, cv=10)

#Fit the model
best_model = clf.fit(x_train,y_train)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

