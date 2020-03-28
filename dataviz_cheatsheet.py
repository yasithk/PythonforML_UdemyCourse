## Exploratory Data Analysis cheat sheet
### Yasith Kariyawasam

## Distribution plots
sns.distplot(tips['total_bill'])
### remove kde layer using
sns.distplot(tips['total_bill'], kde=False)


## Joint Plot
sns.jointplot(x='variable1',y='variable2',data=data,kind='scatter')
## use kind to specify type of jointplot i.e 'kind = kde'

#Pair Plots
sbr.pairplot(df,hue='Clicked on Ad',palette='rainbow')

