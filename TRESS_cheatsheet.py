### Decision Trees and Random Forests

###Tree methods
####Nodes - split for the value of a certain attribute
####Edges outcome of a split to the next node
### Roots
### Leaves

### Entropy and Information gain are the mathematical methods of choosing the best split. Refer to the reading assignment.

### Random forests are used to improve decision trees that uses bagging. Ensemble of decision tress with bootstrapping

### Split data into train and test before this

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train, Y_train)

### Predict on test data
predictions = dtree.Predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test, predictions))


### Random Forests 
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, Y_train)

rfc_pred = rfc.predict(X_test)
