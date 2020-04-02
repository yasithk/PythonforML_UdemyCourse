#### Principal Component Analyis
#####https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643


###The central idea of principal component analysis (PCA) is to reduce the dimensionality of a data set consisting of a large number
###of interrelated variables while retaining as much as possible of the variation present in the data set. This is achieved by transforming
### to a new set of variables, the principal components (PCs), which are uncorrelated, and which are ordered so that the first few retain most
## of the variation present in all of the original variables.


### Principal Component Analysis visualisation
## find what component are the most important ones in a classification
## use for high dimensional data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform()

#PCA
from sklearn.decomposition import PCA
pca = PCA(#spcifiy the number of components you want to keep
	n_components = 2)
pca.fit(scaled_data)

#transform the data into its first PC
x_pca = pca.transform(scaled_data)
scaled_data.shape()
x_pca.shape()

### Plot PC
plt.figure(#size of Plot
	)
####### plot all the columsn
plt.scatter(x_pca[:,0], x_pca[:,1])
plt.xlable('First Principal Component')

#### array of com
pca.components_
##	each row represents a principal component and each columns represent back to the data

### create a dataset with new PCP
df_comp = pd.DataFrame(pca.components_, columns = cancer['feature_names'])