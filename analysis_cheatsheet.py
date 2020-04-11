#Data analysis run sheet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#read in a data set
dataframe_name = pd.read_csv()


#Exploratory Data Analysis

##Missing values analysis
####heat map of boolean values
sns.heatmap(dataframe_name.isnull(), yticklabels=False, cbar=False, cmap= 'viridis')

#histogram
sns.set_style('whitegrid')
sns.countplot(x='variable1',data=dataframe_name,palette='RdBu_r')

## Countplots
sns.Countplot(x = 'variable1', hue = 'variable2', data = dataframe_name, palette='rainbow' )

### describe with groupby 
messages.groupby('label').describe()

### print the entire values of a cell usig iloc
messages[messages['length']== 910]['message'].iloc[0]

#Data Cleansing

## check for missing data
#### count for NULL values per variable
df.isnull().sum()

## filling in missing data
### impute average age per group, define a function then apply over a dataset
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
## apply function over dataset
     train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)   

## drop rows with missing values in a column named Cabin
train.drop('Cabin',axis=1,inplace=True)


## Convert datetime variable. Convert a string into a datetime object
df['date'] = pd.to_datetime(df['date'])
df['date'].head()

## grab the year, month of the date
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)
df.head()

## Standardising data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
### drop the target variable
scaler.fit(dataframe_name.drop('TARGET CLASS',axis=1))
### scaled and transform the variables
scaled_features = scaler.transform(dataframe_name.drop('TARGET CLASS',axis=1))
#### create new feature data frame
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


## Converting Categorical features
#### use pd.get_dummies()
##### drop_first parameter will drop a variable to avoid multi-collinearity 

#create new data frames with dummy variables
sex = pd.get_dummies(train['Sex'], drop_first=True)

#drop the existing categorical data
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

#add the new dummy vars to the existing 
train = pd.concat([train,sex,embark],axis=1)

### merge dataframes
df = pd.merge(df, movie_titles, on ='item_id')

#group by and sort
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

#user pivot_Tablt to get into matrix form
moviemat = df.pivot_table(index= 'user_id', columns='title', values = 'rating')

-----------------------------------------------

#Build a logistic regression model

from sklearn.model_selection import train_test_split

####separate the explanatory variables with the predictor 
X = final_data.drop('not.fully.paid', axis = 1)
Y = final_data['not.fully.paid']
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size = 0.3)


## Training and Predicting 
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
####fit model to datasets
logmodel.fit(X_train,y_train)

#### predict on a test dataset
predictions = logmodel.predict(X_test)

## Evaluation 
#### classification report precision and accuracy - don't need to read off a confusion matrix
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
