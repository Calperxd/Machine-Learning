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
    # we extract the title from each name
    x['Title'] = x['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated title
    # we map each title
    x['Title'] = x.Title.map(Title_Dictionary)
    return x


def process_embarked():
    global x
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    x.Embarked.fillna('S', inplace=True)
    # dummy encoding
    embarked_dummies = pd.get_dummies(x['Embarked'], prefix='Embarked')
    x = pd.concat([x, embarked_dummies], axis=1)
    x.drop('Embarked', axis=1, inplace=True)
    return x


def process_cabin():
    global x
    # replacing missing cabins with U (for Uknown)
    x.Cabin.fillna('U', inplace=True)


    # dummy encoding ...
    cabin_dummies = pd.get_dummies(x['Cabin'], prefix='Cabin')
    x = pd.concat([x, cabin_dummies], axis=1)

    x.drop('Cabin', axis=1, inplace=True)
    return x




x = get_titles()
x = process_embarked()
x = process_cabin()

#############################################################################################################