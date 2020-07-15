#!/usr/bin/env python
# coding: utf-8

# # TPOT using KNN with default hyperparameters 

# ### Auto Dataset

# In[2]:


import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from pmlb import classification_dataset_names
from pmlb import fetch_data


# In[3]:


auto = fetch_data('auto', local_cache_dir='./')
auto.head()


# In[4]:


pd.isnull(auto).any()


# In[5]:


features_auto = auto.drop('target', axis = 1)


# In[6]:


train_auto_features, test_auto_features, train_auto_target, test_auto_target = train_test_split(features_auto, auto['target'], train_size = 0.75, test_size = 0.25, random_state = 42) 


# In[7]:


train_auto_features.size, test_auto_features.size


# In[11]:


#Specify that we want to use KNN 
tpot_config = {
  'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(20,60,101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    }
}


# In[12]:


tpot = TPOTClassifier(generations=100, verbosity=2,
                      config_dict=tpot_config, random_state = 42)
tpot.fit(train_auto_features, train_auto_target)
print(tpot.score(test_auto_features, test_auto_target))


# In[13]:


tpot.fitted_pipeline_


# In[14]:


tpot.export('tpot_KNN_auto.py')


# ### Banana

# In[15]:


banana = fetch_data('banana', local_cache_dir = './')
banana.head()


# In[16]:


pd.isnull(banana).any()


# In[17]:


features_banana = banana.drop('target', axis = 1)


# In[18]:


train_banana_features, test_banana_features, train_banana_target, test_banana_target = train_test_split(features_banana, banana['target'], train_size= 0.75, test_size = 0.25, random_state=42)


# In[19]:


train_banana_features.size, test_banana_features.size


# In[20]:


tpot = TPOTClassifier(generations=100, verbosity=2,
                      config_dict=tpot_config, random_state = 42)
tpot.fit(train_banana_features, train_banana_target)
print(tpot.score(test_banana_features, test_banana_target))


# In[21]:


tpot.export('tpot_KNN_banana.py')


# ### Breast Cancer

# In[22]:


B_Cancer = fetch_data('breast-cancer', local_cache_dir = './')


# In[23]:


B_Cancer.head()


# In[24]:


pd.isnull(B_Cancer).any()


# In[25]:


features_B_Cancer = B_Cancer.drop('target', axis = 1)


# In[26]:


train_cancer_features, test_cancer_features, train_cancer_target, test_cancer_target = train_test_split(features_B_Cancer , B_Cancer['target'], train_size = 0.75, test_size = 0.25 , random_state =42 )


# In[27]:


train_cancer_features.size, test_cancer_features.size


# In[ ]:


tpot = TPOTClassifier(generations=100, verbosity=2,
                      config_dict=tpot_config, random_state = 42)
tpot.fit(train_cancer_features, train_cancer_target)


# In[ ]:


print(tpot.score(test_cancer_features, test_cancer_target))


# In[ ]:


tpot.export('tpot_KNN_Breast_Cancer.py')


# ### Contraceptive

# In[ ]:


contraceptive = fetch_data('contraceptive', local_cache_dir='./')


# In[ ]:


contraceptive.head()


# In[ ]:


pd.isnull(contraceptive).any()


# In[ ]:


features_contraceptive = contraceptive.drop('target', axis =1)


# In[ ]:


train_contra_features, test_contra_features, train_contra_target, test_contra_target = train_test_split(features_contraceptive, contraceptive['target'], train_size=0.75, test_size = 0.25, random_state=42)


# In[ ]:


train_contra_features.size, test_contra_features.size


# In[ ]:


tpot = TPOTClassifier(generations=100, verbosity=2,
                      config_dict=tpot_config, random_state = 42)
tpot.fit(train_contra_features, train_contra_target)
print(tpot.score(test_contra_features, test_contra_target))


# In[ ]:


tpot.export('tpot_KNN_contraception.py')


# ### Pima

# In[ ]:


pima = fetch_data('pima', local_cache_dir = './')
pima.head()


# In[ ]:


pd.isnull(pima).any()


# In[ ]:


features_pima = pima.drop('target', axis=1)


# In[ ]:


train_pima_features, test_pima_features, train_pima_target, test_pima_target = train_test_split(features_pima, pima['target'], train_size=0.75, test_size=0.25, random_state=42)


# In[ ]:


tpot = TPOTClassifier(generations=100, verbosity=2,
                      config_dict=tpot_config, random_state = 42)
tpot.fit(train_pima_features, train_pima_target)
print(tpot.score(test_pima_features, test_pima_target))


# In[ ]:


tpot.export('tpot_KNN_pima.py')

