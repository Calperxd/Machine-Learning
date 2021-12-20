import pandas as pd

############################################################################################################################################################################################
# Writing to csv files
# Creating a new dataframe to max values

df = pd.read_json('max.2n3po9nq.json')
training = pd.DataFrame(columns=['x','y','z','class'])                       # Dataframe
for i in range(len(df.loc[ "values", :][2])):
    training.loc[i,'x'],training.loc[i,'y'],training.loc[i,'z'] = df.loc[ "values", :][2][i][0],df.loc[ "values", :][2][i][1],df.loc[ "values", :][2][i][2]

training['class'] = 0
training.to_csv('data.csv', mode='a')
############################################################################################################################################################################################
# Writing to csv files
# Creating a new dataframe to mid values


df = pd.read_json('mid.2n3peauo.json')
training = pd.DataFrame(columns=['x','y','z','class'])                       # Dataframe
for i in range(len(df.loc[ "values", :][2])):
    training.loc[i,'x'],training.loc[i,'y'],training.loc[i,'z'] = df.loc[ "values", :][2][i][0],df.loc[ "values", :][2][i][1],df.loc[ "values", :][2][i][2]

training['class'] = 1
training.to_csv('data.csv',mode='a', header='none')
############################################################################################################################################################################################
# Writing to csv files
# Creating a new dataframe to low values


df = pd.read_json('low.2n3p0qj3.json')
training = pd.DataFrame(columns=['x','y','z','class'])                       # Dataframe
for i in range(len(df.loc[ "values", :][2])):
    training.loc[i,'x'],training.loc[i,'y'],training.loc[i,'z'] = df.loc[ "values", :][2][i][0],df.loc[ "values", :][2][i][1],df.loc[ "values", :][2][i][2]


training['class'] = 2
training.to_csv('data.csv', mode='a', header='none')

