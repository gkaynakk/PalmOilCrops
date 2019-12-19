# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:37:22 2019

@author: Gurcan Kaynak
"""
#import pandas for importing csv files 
import pandas as pd
#Encoding 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Missing data treatment
from sklearn.preprocessing import Imputer
#Plotting
import matplotlib.pyplot as plt
#Scatter plot
import seaborn as sns
#feature importance
from sklearn.ensemble import ExtraTreesClassifier
#Regression Models
from sklearn.linear_model import LinearRegression
#Training and testing of our data
from sklearn.model_selection import train_test_split

  
#Getting data set
df = pd.read_excel (r'C:/Data Science/Machine Learning A-Z Template Folder/PalmOilCrops/CropCompute.xls')

#==============================================================================
#The data can be categorised into predictors (features) and the response variable. 
#The dependent variables are Total and Yield Per Hectare (YPH)
#YPH is a better measure of productivity and we choose it as our response variable.

#Using prediction techniques to analyse what factors affect the YPH

#Dataset Preprocessing: Getting the dataset to an easier-to-use form for 
#visualisation and modelling

#Renaming columns (removing spaces and making shorter column names for ease)
df.rename(columns={'Area (Hectare)':'Area',
                         'Field Type':'FType',
                         'Harvesting Year':'HYear',
                         'Year of Planting':'PYear',
                         'Yield Per Hectare (Ton)':'YPH',
                         'Annual Crop (Ton)':'TotalCrop'
                         }, inplace=True) 

print("Names of columns after renaming:\n")
for col in df.columns: 
    print(col)
    #This gives the values of the dataframe
    X = df.iloc[:, :-1].values
    #X is not a dataframe It is an array.
#==============================================================================
#Filling Missing Data: Imputer

#Encoding categorical data:  
#All the categorical variables we have are nominal, and not ordered

#Label encoding converts categorical variables into numbers     
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#In order to not to confuse the model into thinking that 
#Column data has hierarchy so we do one hot encoding 
onehotencoder = OneHotEncoder(categorical_features = [0])
onehotencoder = onehotencoder.fit(X)
X=onehotencoder.transform(X)
X=X.toarray()
dfencoded=pd.DataFrame(X)

#==============================================================================
#handling missing values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#Train the imputor on the  dataset
imputer = imputer.fit(X)
#X.values attribute return a Numpy representation of X.
#Apply the imputer to the df dataset
X_transformed= imputer.transform(X)

#Visualization of datasets using boxplot
AGE=df['Age']
AREA=df['Area']
yph=df['YPH']


box_plot_data=[AGE,AREA]
box=plt.boxplot(box_plot_data,patch_artist=True,labels=['AGE','AREA']) 

colors = ['cyan', 'lightblue', 'lightgreen', 'tan','purple']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.show()
#==============================================================================
#Determining the trend between the predictors and the response variable.
g =sns.scatterplot(x="Area", y="YPH",
              data=df, 
              legend='full')

plt.title('Does high plantation area yield more palm oil crop?')
plt.show()
#We see that the highest productivity comes from smaller blocks 
#A decline is seen as the area keeps increasing.

#==============================================================================
#To further investigate the effect of age on productivity, we group the data by planting year
#and look at particular subsets. 
#We plot the productivity of the different blocks planted in the same year 
#Going through the same part of their life cycle (age-wise)
# 
#Each colour represents a different block. Each line shows how the productivity
#of each block has been changing as the age increases

df_blocks=df[df.PYear == 2013]

for name, data in df_blocks.groupby('Block'):
    plt.plot(data['Age'], data['YPH'], label=name)

plt.title('Age vs Yield Per Hectare for Blocks Planted in 2013 (Age 1-5)')
plt.xlabel('Age')
plt.ylabel('YPH')
plt.legend()
plt.show()
#==============================================================================
df_blocks=df[df.PYear == 2007]

for name, data in df_blocks.groupby('Block'):
    plt.plot(data['Age'], data['YPH'], label=name)

plt.title("Age vs Yield Per Hectare for Blocks Planted in 2007 (Age (7-11))")
plt.xlabel('Age')
plt.ylabel('YPH')
plt.legend()
plt.show()
#==============================================================================
df_blocks=df[df.PYear == 1996]

for name, data in df_blocks.groupby('Block'):
    plt.plot(data['Age'], data['YPH'], label=name)

plt.title("Age vs Yield Per Hectare for Blocks Planted in 1996 (Age (18-22))")
plt.xlabel('Age')
plt.ylabel('YPH')
plt.legend()
plt.show()
#==============================================================================
df_blocks=df[df.PYear == 1991]

for name, data in df_blocks.groupby('Block'):
    plt.plot(data['Age'], data['YPH'], label=name)

plt.title("Age vs Yield Per Hectare for Blocks Planted in 1991 (Age 23-27)")
plt.xlabel('Age')
plt.ylabel('YPH')
plt.legend()
plt.show()
#==============================================================================
#After 25 years of age, we see that there is certainly a decline in yield in all the blocks, 
#but the exact trends of this decline does not seem like it can be explained by Age. 
#Now, that we have a vague understanding of the trends and relationships using the graphs, 
#we move to actually building our model.

#Building a Model
#Feature Selection
#Creating and training a model is dependent purely on the data that it receives. 
#Putting in irrelevant data returns a worse model.
#This is also called GIGO. (Garbage In, Garbage Out).
#===============================================================
#Feature Importance we can use 3 different methods
#Method1:Univariate Selection
X1 = dfencoded.iloc[:,0:9]  #independent columns
y1 = dfencoded.iloc[:,-1]    #target column
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#uses the chi-squared (chi²) statistical test for non-negative features to select
#10 of the best features 
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=3)
fit = bestfeatures.fit(X1,y1)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X1.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
print("\n0:Division, 1:Block, 2:Area, 3:Field Type, 4:Harvesting Year\n5:Year of Planting, 6:Age, 7: Total Crop, 8:Yield Per Hectare\n")
#==============================================================================
model = ExtraTreesClassifier()
model.fit(X1,y1)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
print("0:Division, 1:Block, 2:Area, 3:Field Type, 4:Harvesting Year\n5:Year of Planting, 6:Age, 7: Total Crop, 8:Yield Per Hectare\n")
#returns a vector of the features that the algorithm has deemed important 
#=============================================================================

#Predictive Data Analytics
#Splitting the dataset into the Training set and Test set
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2,random_state=0)
X1_train.head()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X1_train = sc_X.fit_transform(X1_train)
X1_test = sc_X.transform(X1_test)
sc_y = StandardScaler()
y1_train = sc_y.fit_transform(y1_train.values.reshape(-1,1))


regressor = LinearRegression()
regressor.fit(X1_train, y1_train)

#Predicting the Test set results
y_pred = regressor.predict(X1_test)

import statsmodels.api as sm
X2 = sm.add_constant(X)
est = sm.OLS(y1, X1)
est2 = est.fit()
print(est2.summary())

#Mean Square Error and Root Mean Square Error
from sklearn.metrics import mean_squared_error 
from math import sqrt
meanSquaredError=mean_squared_error(y1_test, y_pred )
print("MSE:", meanSquaredError)
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)
print('Model Accuracy with linear regression:', regressor.score(X1,y1))
#=============================================================================
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressorTree = DecisionTreeRegressor(random_state = 0)
regressorTree.fit(X1_train, y1_train)

# Predicting a new result
y_pred1 = regressorTree.predict(X1_test)

print('Decision Tree Regression Accuracy:',regressorTree.score(X1,y1))
