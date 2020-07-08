#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split


# In[4]:


covid_train = pd.read_csv('https://raw.githubusercontent.com/trang1618/Pre_Surv_COVID_19/c3bf4d146ca50e4e5de3c9d555547345e02d7698/data/processed_covid_train.tsv',
                    sep = '\t')
covid_test = pd.read_csv('https://raw.githubusercontent.com/trang1618/Pre_Surv_COVID_19/master/data/processed_covid_test.tsv' , sep = '\t')


# In[5]:


covid_train.head()


# In[6]:


covid_test.head()


# In[ ]:




