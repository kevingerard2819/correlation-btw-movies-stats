#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)


# In[2]:


df=pd.read_csv('movies.csv')


# In[3]:


df.head()


# In[4]:


df.dropna(inplace=True)


# In[5]:


df.dtypes


# In[6]:


df['budget']=df['budget'].astype('int64')
df['gross']=df['gross'].astype('int64')


# In[7]:


df.head()


# In[8]:


#Sorting movies in descending order by gross
df=df.sort_values(by='gross',ascending=False,inplace=False)


# In[9]:


df.head()


# In[10]:


pd.set_option('display.max_rows',None)


# In[11]:


plt.scatter(x=df['budget'],y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget')
plt.ylabel('Gross')


# In[12]:


df.head()


# In[13]:



#Regression plot between budget and gross earnings
sns.regplot(x='budget',y='gross',data=df,line_kws={'color':'blue'})


# In[14]:


correlation_matrix=df.corr('pearson') 
correlation_matrix


# In[15]:


sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.title('Correlation matrix for numerical movie features')
plt.xlabel('Movie features')
plt.ylabel('Movie features')


# In[16]:


#Converting object-type columns to numericals
df_numerized=df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype=='object'):
        df_numerized[col_name]=df_numerized[col_name].astype('category')
        df_numerized[col_name]=df_numerized[col_name].cat.codes
df_numerized.head()


# In[17]:


correlation_matrix=df_numerized.corr('pearson')
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.title('Correlation matrix for numerical movie features')
plt.xlabel('Movie features')
plt.ylabel('Movie features')


# In[18]:


correlation_map=df_numerized.corr()
corr_pair=correlation_map.unstack()
corr_pair


# In[19]:


sorted_pairs=corr_pair.sort_values(ascending=False)
sorted_pairs


# In[20]:


#Movie statistics which are highly correlated
high_corr=sorted_pairs[(sorted_pairs)>0.4]
high_corr


# In[ ]:




