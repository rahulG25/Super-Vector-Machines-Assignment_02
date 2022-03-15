#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('forestfires.csv')
df


# In[3]:


#Checking for null values & data types
df.info()


# In[4]:


#Scaling the data (leaving out the target variable, and the taking only the numerical data for input)
df1= df.iloc[:,2:30]


# # Since the no. of columns are more here we will need to apply PCA

# Applyting Dimentionality Reduction technique PCA

# In[5]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(df1)
df_norm = sc.transform(df1)
df_norm                     #Normalised dataset


# In[6]:


from sklearn.decomposition import PCA

pca = PCA(n_components = 28)
pca_values = pca.fit_transform(df_norm)
pca_values


# In[7]:


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var


# In[8]:


# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[9]:


import matplotlib.pyplot as plt

# Variance plot for PCA components obtained
plt.figure(figsize=(12,4))
plt.plot(var1);


# # Let's select 1st 25 columns for model creation, as looking at the data varience we understand that we get 99.91% of the data in 1st 25 columns
# 

# In[10]:


finalDf = pd.concat([pd.DataFrame(pca_values[:,0:24],columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7',
                                                             'pc8','pc9','pc10','pc11','pc12','pc13','pc14',
                                                             'pc15','pc16','pc17','pc18','pc19','pc20','pc21',
                                                             'pc22','pc23','pc24']),
                     df[['size_category']]], axis = 1)
finalDf


# In[11]:


array = finalDf.values
X = array[:,0:24]
Y = array[:,24]


# # SVM Classification

# Let's use Grid search CV to find out best value for params

# In[12]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],'C':[1,10,100,1000] },
             {'kernel':['linear'],'C':[1,10,100,1000]}]
gsv = GridSearchCV(clf,param_grid,cv=10,n_jobs=-1)
gsv.fit(X,Y)

gsv.best_params_ , gsv.best_score_


# In[14]:


clf = SVC(C=100, kernel='linear')
results = cross_val_score(clf, X, Y, cv=10)

print(results.mean())

