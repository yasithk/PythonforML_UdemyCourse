### Support Vector Machines cheat sheet


### train, test split data
from sklearn.model_selection import train_test_split

X= df_feat
Y=cancer['target']

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

### SVM model training
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

#Gridsearch
#Finding the right parameters (like what C or gamma values to use) is a tricky task!
# But luckily, we can be a little lazy and just try a bunch of combinations and see what works best!
# This idea of creating a 'grid' of parameters and just trying out all the possible combinations is called a Gridsearch, this method is common enough that Scikit-learn has this functionality built in with GridSearchCV! The CV stands for cross-validation which is the
#GridSearchCV takes a dictionary that describes the parameters that should be tried and a model to train. The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested.

#C controls the cost of misclassification on training data. Large c value = low bias and high variance because we penalise the cost of misclassification a lot
#Low c value = high bias but lower variance
#Gamma parameter = free parameter in the'radiab basi function'. Large gamma value lead to high bias low variance. 
#https://medium.com/@ankitnitjsr13/math-behind-support-vector-machine-svm-5e7376d0ee4d

from sklearn.grid_search import GridSearchCV

param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)

grid.fit(X_train, Y_train)
#fit runs the same loop for cross-validation to find best parameter combinations, then runs again without cross validation to build new model

#find the parameters
grid.best_params_ 

gird.best_estimator

