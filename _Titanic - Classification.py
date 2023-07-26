#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we target to complete the analysis of what sorts of people were likely to survive.
# 
# https://www.kaggle.com/c/titanic/data

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

sns.set(rc={'figure.figsize':(12, 10)})


# ## Loading Dataset

# In[2]:


data = pd.read_csv('C://Users//Pranjal//OneDrive//Desktop//data_analyst//csv file//titanic_data.csv')
data.head(10)


# ## Types of Features :
# 
# Categorical - Sex, and Embarked.
# 
# Continuous - Age, Fare
# 
# Discrete - SibSp, Parch.
# 
# Alphanumeric - Cabin

# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data.describe()


# ## Numerical Value Analysis

# In[7]:


plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(data[["Survived","SibSp","Parch","Age","Fare"]].corr(), annot=True)


# ## Conclusion :
# 
# Only Fare feature seems to have a significative correlation with the survival probability.
# 
# It doesn't mean that the other features are not usefull. Subpopulations in these features can be correlated with the survival. To determine this, we need to explore in detail these features

# ## sibsp - Number of siblings / spouses aboard the Titanic

# In[8]:


data['SibSp'].nunique()


# In[9]:


data['SibSp'].unique()


# In[10]:


bargraph_sibsp = sns.factorplot(x = "SibSp", y = "Survived", data = data, kind = "bar", size = 8)
bargraph_sibsp = bargraph_sibsp.set_ylabels("survival probability")


# It seems that passengers having a lot of siblings/spouses have less chance to survive.
# 
# 
# Single passengers (0 SibSP) or with two other persons (SibSP 1 or 2) have more chance to survive.

# ## Age

# In[11]:


age_visual = sns.FacetGrid(data, col = 'Survived', size=7)
age_visual = age_visual.map(sns.distplot, "Age")
age_visual = age_visual.set_ylabels("survival probability")


# Age distribution seems to be a tailed distribution, maybe a gaussian distribution.
# 
# We notice that age distributions are not the same in the survived and not survived subpopulations. Indeed, there is a peak corresponding to young passengers, that have survived. We also see that passengers between 60-80 have less survived.
# 
# So, even if "Age" is not correlated with "Survived", we can see that there is age categories of passengers that of have more or less chance to survive.
# 
# It seems that very young passengers have more chance to survive.

# ## Sex

# In[12]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 10))
age_plot = sns.barplot(x = "Sex",y = "Survived", data = data)
age_plot = age_plot.set_ylabel("Survival Probability")


# In[13]:


data[["Sex","Survived"]].groupby('Sex').mean()


# It is clearly obvious that Male have less chance to survive than Female. So Sex, might play an important role in the prediction of the survival. For those who have seen the Titanic movie (1997), I am sure, we all remember this sentence during the evacuation - Women and children first

# ## PClass

# In[14]:


pclass = sns.factorplot(x = "Pclass", y = "Survived", data = data, kind = "bar", size = 8)
pclass = pclass.set_ylabels("survival probability")


# ## Pclass vs Survived by Sex

# In[15]:


g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=data, size=6, kind="bar")
g = g.set_ylabels("survival probability")

import warnings
warnings.filterwarnings("ignore")


# ## Embarked

# In[16]:


data["Embarked"].isnull().sum()


# In[17]:


data["Embarked"].value_counts()


# In[18]:


#Fill Embarked with 'S' i.e. the most frequent values
data["Embarked"] = data["Embarked"].fillna("S")


# In[19]:


g = sns.factorplot(x="Embarked", y="Survived", data=data, size=7, kind="bar")
g = g.set_ylabels("survival probability")


# Passenger coming from Cherbourg (C) have more chance to survive.

# ## Let's find the reason

# In[21]:


# Explore Pclass vs Embarked 
g = sns.factorplot("Pclass", col="Embarked",  data=data, size=7, kind="count")
g.despine(left=True)
g = g.set_ylabels("Count")


# In[22]:


g = sns.factorplot("Sex", col="Embarked",  data=data, size=7, kind="count")


# Cherbourg passengers are mostly in first class which have the highest survival rate.
# 
# 
# Southampton (S) and Queenstown (Q) passangers are mostly in third class.

# ## Preparing data

# In[23]:


data = pd.read_csv('C://Users//Pranjal//OneDrive//Desktop//data_analyst//csv file//titanic_data.csv')
data.head(10)


# In[25]:


data.info()


# In[26]:


mean = data["Age"].mean()
std = data["Age"].std()
is_null = data["Age"].isnull().sum()
    
# compute random numbers between the mean, std and is_null
rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    
# fill NaN values in Age column with random values generated
age_slice = data["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
data["Age"] = age_slice


# In[27]:


data["Age"].isnull().sum()


# In[28]:


data.info()


# In[29]:


data["Embarked"].isnull().sum()


# In[30]:


#Fill Embarked with 'S' i.e. the most frequent values
data["Embarked"] = data["Embarked"].fillna("S")


# In[31]:


col_to_drop = ['PassengerId','Cabin', 'Ticket','Name']
data.drop(col_to_drop, axis=1, inplace = True)


# In[32]:


data.head()


# In[33]:


genders = {"male": 0, "female": 1}
data['Sex'] = data['Sex'].map(genders)


# In[34]:


data.head()


# In[35]:



ports = {"S": 0, "C": 1, "Q": 2}

data['Embarked'] = data['Embarked'].map(ports)


# In[36]:


data.head()


# In[37]:


data.info()


# ## Splitting data

# In[38]:


# input and output data

x = data.drop(data.columns[[0]], axis = 1)
y = data['Survived']


# In[39]:


x.head()


# In[40]:


y.head()


# In[41]:



# splitting into training and testing data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.30, random_state =0)


# ## Feature Scaling

# In[42]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain) 
xtest = sc_x.transform(xtest)


# ## Classification

# In[43]:


logreg = LogisticRegression()
svc_classifier = SVC()
dt_classifier = DecisionTreeClassifier()
knn_classifier = KNeighborsClassifier(5)
rf_classifier = RandomForestClassifier(n_estimators=1000, criterion = 'entropy', random_state = 0 )


# In[44]:


logreg.fit(xtrain, ytrain)
svc_classifier.fit(xtrain, ytrain)
dt_classifier.fit(xtrain, ytrain)
knn_classifier.fit(xtrain, ytrain)
rf_classifier.fit(xtrain, ytrain)


# In[45]:


logreg_ypred = logreg.predict(xtest)
svc_classifier_ypred = svc_classifier.predict(xtest)
dt_classifier_ypred = dt_classifier.predict(xtest)
knn_classifier_ypred = knn_classifier.predict(xtest)
rf_classifier_ypred = rf_classifier.predict(xtest)


# In[46]:


# finding accuracy
from sklearn.metrics import accuracy_score

logreg_acc = accuracy_score(ytest, logreg_ypred)
svc_classifier_acc = accuracy_score(ytest, svc_classifier_ypred)
dt_classifier_acc = accuracy_score(ytest, dt_classifier_ypred)
knn_classifier_acc = accuracy_score(ytest, knn_classifier_ypred)
rf_classifier_acc = accuracy_score(ytest, rf_classifier_ypred)


# In[47]:


print ("Logistic Regression : ", round(logreg_acc*100, 2))
print ("Support Vector      : ", round(svc_classifier_acc*100, 2))
print ("Decision Tree       : ", round(dt_classifier_acc*100, 2))
print ("K-NN Classifier     : ", round(knn_classifier_acc*100, 2))
print ("Random Forest       : ", round(rf_classifier_acc*100, 2))


# In[ ]:




