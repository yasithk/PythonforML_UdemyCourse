# K nearest neighbours

##Pros
#### very simple
### training is trivial
### Few parameters - K, distance

##Cons
### high prediction cost (Worse for large datasets)
### not good with high dimensional data
### categorical variables don't work well 

##*** SPLIT DATA BEFORE THE BELOW SETS**###

## Choosing a K value
error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    ## average error rate and append to actuals
    error_rate.append(np.mean(pred_i != y_test))

#### Plot KNN tree
 plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')


### Using KNN
from sklearn.neighbors import KNeighborsClassifier
### knn is the name given to our knn algorithm
knn = KNeighborsClassifier(n_neighbors=1)
### fit to train data
knn.fit(X_train,y_train)
### pred to test data
pred = knn.predict(X_test)

## evaluation 
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,pred))
