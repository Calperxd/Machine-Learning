import numpy as np
import pandas as pd



#loading the data
train_data = pd.read_csv("train.csv", sep=";")

#############################################################################################################
#pre processing
train_data = train_data.dropna(subset=['Age'])                                  #removendo linhas da idade que são NaN
train_data['Cabin'] = train_data['Cabin'].fillna(0)                             #Zerando os que não possuem cabine
train_data = train_data.dropna(subset=['Embarked'])                             #Removendo os que não estão cadastrados como embarcados


x = train_data.iloc[:, 2:]                                          #Features
y = train_data.iloc[:, 1]                                           #Classe vivo ou morto

titles = set()
for name in x['Name']:
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
    x['Title'] = x['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    x['Title'] = x.Title.map(Title_Dictionary)
    return x


def process_embarked():
    global x
    x.Embarked.fillna('S', inplace=True)
    embarked_dummies = pd.get_dummies(x['Embarked'], prefix='Embarked')
    x = pd.concat([x, embarked_dummies], axis=1)
    x.drop('Embarked', axis=1, inplace=True)
    return x


def process_sex():
    global x
    x['Sex'] = x['Sex'].map({'male':1, 'female':0})
    return x


def process_pclass():
    
    global x
    pclass_dummies = pd.get_dummies(x['Pclass'], prefix="Pclass")
    x = pd.concat([x, pclass_dummies],axis=1)
    x.drop('Pclass',axis=1,inplace=True)
    return x


def process_ticket():
    global x
    class_mapping = {label: idx for idx, label in enumerate(np.unique(x['Ticket']))}
    x['Ticket'] = x['Ticket'].map(class_mapping)
    return x


def process_family():
    global x
    x['FamilySize'] = x['Parch'] + x['SibSp'] + 1
    x['Singleton'] = x['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    x['SmallFamily'] = x['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    x['LargeFamily'] = x['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    return x


def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


def recover_train_test_target():
    global x
    
    targets = pd.read_csv('./data/train.csv', usecols=['Survived'])['Survived'].values
    train = x.iloc[:891]
    test = x.iloc[891:]
    
    return train, test, targets





x = get_titles()
x = process_embarked()
x = process_sex()
x = process_ticket()

train, test, targets = recover_train_test_target()
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25, 25))
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
print(train_reduced.shape)
# (891L, 14L)

test_reduced = model.transform(test)
print(test_reduced.shape)
# (418L, 14L)




print(x.head(15))

#############################################################################################################