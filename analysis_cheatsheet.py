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


#Data Cleansing

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

## Converting Categorical features
#### use pd.get_dummies()
##### drop_first parameter will drop a variable to avoid multi-collinearity 

#create new data frames with dummy variables
sex = pd.get_dummies(train['Sex'], drop_first=True)

#drop the existing categorical data
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

#add the new dummy vars to the existing 
train = pd.concat([train,sex,embark],axis=1)

-----------------------------------------------

#Build a logistic regression model

from sklearn.model_selection import train_test_split

####separate the explanatory variables with the predictor 
X = train.drop('Survived', axis=1)
y = train['Survived']

###split the dataset
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], test_size=0.30, 
                                                    random_state=101)

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
