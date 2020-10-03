#!/usr/bin/env python
# coding: utf-8

# # Manali Kulkarni

# # Task-4: To Explore Decision Tree Algorithm.
#      For the given iris dataset, I have created the decision tree classifier and visualised it graphically.The purpose is if we feed any new data to this classifier,it would be able to predict the right class accordingly.

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import datasets


# In[13]:


#Loading the dataset
iris=datasets.load_iris()
iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_data.head(5)


# In[14]:


y=iris.target
print(y)


# In[16]:


#summerising
iris_data.describe()


# In[18]:


#Correlation Matrix Plot
CorrMatrix=iris_data.corr()
print(CorrMatrix)
sn.heatmap(CorrMatrix,annot=True)
plt.show()


# In[20]:


sn.pairplot(iris_data)


# In[21]:


#Splitting the data into training & testing sets
x=iris_data
y=y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
print("Shape of feature training data:",x_train.shape)
print("Shape of feature training data:",y_train.shape)
print("Shape of feature test data:",x_test.shape)
print("Shape of feature test data:",y_test.shape)


# In[22]:


#Applying decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[23]:


plt.figure(figsize=(20,20))
tree.plot_tree(clf,filled = True,rounded=True,proportion=True,feature_names=iris.feature_names)
plt.show()


# In[ ]:


Conclusion:We have predicted the right class accordingly.


# # Thank You!!
