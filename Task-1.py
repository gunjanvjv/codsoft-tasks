#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# In[3]:


df= pd.read_csv("tested.csv")
df


# In[4]:


df.info


# In[6]:


df.describe()


# In[7]:


df.columns


# In[12]:


df_num = df[['Age','SibSp','Parch','Fare']]
df_num


# In[13]:


df_cat = df[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
df_cat


# In[14]:


for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


# In[15]:


sns.heatmap(df_num.corr())


# In[17]:


pd.pivot_table(df, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])
 


# In[18]:


for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    plt.show()


# In[20]:


print(pd.pivot_table(df, index = 'Survived', columns = 'Pclass',
                     values = 'Ticket' ,aggfunc ='count'))
print()
print(pd.pivot_table(df, index = 'Survived', columns = 'Sex', 
                     values = 'Ticket' ,aggfunc ='count'))
print()
print(pd.pivot_table(df, index = 'Survived', columns = 'Embarked', 
                     values = 'Ticket' ,aggfunc ='count'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




