#loading the data
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#############################################################################################################
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


def process_fare():
    global train_data
    train_data.Fare.fillna(train_data.iloc[:891].Fare.mean(), inplace=True)
    return train_data


train_data = get_titles()
train_data = process_pclass()
train_data = process_embarked()
train_data = process_sex()
train_data = process_ticket()
train_data = process_fare()

#pre processing
train_data = train_data.dropna(subset=['Age'])                                  #removendo linhas da idade que são NaN
train_data['Cabin'] = train_data['Cabin'].fillna(0)                             #Zerando os que não possuem cabine



x = train_data.iloc[:, 3:]                                                 #Features
y = train_data[["Survived"]]                                           #Classe vivo ou morto
print(x)

#clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
#clf = clf.fit(x, y)
#features = pd.DataFrame()                                           #Definindo features
#features['feature'] = x.columns
#features['importance'] = clf.feature_importances_
#features.sort_values(by=['importance'], ascending=True, inplace=True)
#features.set_index('feature', inplace=True)
#features.plot(kind='barh', figsize=(25, 25))


#train = combined.iloc[:891]
#test = combined.iloc[891:]

#print(x)
#print(y)


#############################################################################################################