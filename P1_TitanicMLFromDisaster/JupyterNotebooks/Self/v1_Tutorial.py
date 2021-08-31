#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# In[4]:


get_ipython().system('pwd')


# # Loading Data

# In[9]:


train_data = pd.read_csv('/Users/anuja/Anuja/IdeaProjects/Kaggle/P1_TitanicMLFromDisaster/Data/train.csv')
train_data.head()


# In[10]:


test_data = pd.read_csv('/Users/anuja/Anuja/IdeaProjects/Kaggle/P1_TitanicMLFromDisaster/Data/test.csv')
test_data.head()


# In[50]:


features = ["Pclass","Sex","SibSp","Parch","Fare"]
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
y_train = train_data["Survived"]


# # Applying model directly

# In[51]:


log_reg = LogisticRegression()


# In[52]:


log_reg.fit(X_train, y_train)


# In[56]:


X_test['Fare'].fillna(X_test['Fare'].mean(), inplace = True)
X_test.isnull().sum()


# In[61]:


predictions = log_reg.predict(X_test)
Submission = pd.DataFrame({'PassengerID': test_data.PassengerId, 'Survived': predictions})
Submission


# In[63]:


Submission.to_csv('/Users/anuja/Anuja/IdeaProjects/Kaggle/P1_TitanicMLFromDisaster/JupyterNotebooks/Self/submission.csv', index=False)


# In[ ]:




