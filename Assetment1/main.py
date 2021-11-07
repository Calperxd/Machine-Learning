import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer                            #Biblioteca para alterar valores NaN


#loading the data
train_data = pd.read_csv("train.csv", sep=";")

#############################################################################################################
#pre processing
print(train_data.isnull().sum())
train_data = train_data.dropna(subset=['Age'])                                  #removendo linhas da idade que são NaN
imr = SimpleImputer(missing_values='NaN', strategy= 'mean')
imr = imr.fit(train_data)
print(train_data.isnull().sum())
#imr = imr.fit(df.values)
#imputed_data = imr.transform(df.values)
#imputed_data

#print(train_data.isnull().sum())                                               #Checando se a linha das idades foram removidas
#print('Survived', np.unique(train_data['Survived']))
#############################################################################################################



id = train_data.iloc[: , 0]
x = train_data.iloc[:, 2:]                                          #Features
y = train_data.iloc[:, 1]                                           #Classe vivo ou morto




