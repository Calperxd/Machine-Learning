#loading the data
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
import pandas as pd


#############################################################################################################
#pre processing
train_data = pd.read_csv("train.csv", sep=";")


titles = set()
for name in train_data['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

def get_titles():
    train_data['Title'] = train_data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    train_data['Title'] = train_data.Title.map(Title_Dictionary)
    return train_data


def process_embarked():
    global train_data
    train_data.Embarked.fillna('S', inplace=True)
    embarked_dummies = pd.get_dummies(train_data['Embarked'], prefix='Embarked')
    train_data = pd.concat([train_data, embarked_dummies], axis=1)
    train_data.drop('Embarked', axis=1, inplace=True)
    return train_data


def process_sex():
    global train_data
    train_data['Sex'] = train_data['Sex'].map({'male':1, 'female':0})
    return train_data


def process_pclass():
    global train_data
    pclass_dummies = pd.get_dummies(train_data['Pclass'], prefix="Pclass")
    train_data = pd.concat([train_data, pclass_dummies],axis=1)
    train_data.drop('Pclass',axis=1,inplace=True)
    return train_data


def process_ticket():
    global train_data
    class_mapping = {label: idx for idx, label in enumerate(np.unique(train_data['Ticket']))}
    train_data['Ticket'] = train_data['Ticket'].map(class_mapping)
    return train_data


def process_family():
    global train_data
    train_data['FamilySize'] = train_data['Parch'] + train_data['SibSp'] + 1
    train_data['Singleton'] = train_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    train_data['SmallFamily'] = train_data['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    train_data['LargeFamily'] = train_data['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    return train_data



def process_cabin():
    global train_data
    train_data['Cabin'] = train_data['Cabin'].fillna(0)                     # Zerando os que não possuem cabine
    class_le = LabelEncoder()
    train_data = train_data.apply(lambda col: class_le.fit_transform(col.astype(str)), axis=0, result_type='expand')
    x = class_le.fit_transform(train_data['Cabin'].values)
    return train_data

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


train_data = get_titles()
train_data = process_pclass()
train_data = process_embarked()
train_data = process_sex()
train_data = process_ticket()
train_data = process_family()
train_data = process_cabin()

train_data = train_data.dropna(subset=['Age'])                                  #removendo linhas da idade que são NaN
train_data.drop('SibSp', inplace=True, axis=1)                                  #Excluindo coluna de SibSp
train_data.drop('Parch', inplace=True, axis=1)                                  #Excluindo coluna de Parch


x = train_data.iloc[:, 3:]                                                                #Features
y = train_data['Survived'].values                                                         #Classe vivo ou morto

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
models = [clf]
clf.fit(x, y)

for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=x, y=y, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')

# turn run_gs to True if you want to run the gridsearch again.
run_gs = True

if run_gs:
    parameter_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                               )

    grid_search.fit(x, y)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

else:
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50,
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

    model = RandomForestClassifier(**parameters)
    model.fit(x, y)
test = pd.DataFrame({'col_name': clf.feature_importances_}, index=x.columns).sort_values(by='col_name', ascending=False)
#############################################################################################################