#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
df = pd.read_csv("train.csv")


# In[18]:


#Print first 5 rows
df.head()


# In[19]:


#Print last 5 rows
df.tail()


# In[20]:


#Obtain data on the different columns of the dataset
df.describe()


# In[23]:


#Sort dataset in alphabetical order of Name
#na_position helps to set a position for NaN values in the column 'Name'
SortNM = df.sort_values("Name", ascending=True, na_position='last')
SortNM.head()


# In[28]:


#Check count of passengers
df["PassengerId"].shape[0]


# In[40]:


#Check count of different values in column 'Pclass'
df["Pclass"].value_counts()


# In[41]:


#Check count of different values in column 'Survived'
df["Survived"].value_counts()


# In[27]:


#Check count of different values in column 'Embarked'
df["Embarked"].value_counts()


# In[31]:


#Check unique number of tickets out of 891 passengers
df["Ticket"].nunique()


# In[36]:


#Check unique number of Passenger classes and Sex
df[["Pclass","Sex"]].nunique()


# In[38]:


#Shows records where Embarked status is 'S'
EmbS = df["Embarked"]=='S'
df[EmbS].head()


# In[39]:


#Number of Female passengers
Fem = df["Sex"]=="female"
df[Fem].shape[0]


# In[49]:


#Number of passengers who Survived
Surv = df["Survived"]==1
df[Surv].shape[0]


# In[60]:


#Number of female passengers who survived
df[Fem & Surv].shape[0]


# In[48]:


#Number of passengers with Fare>100 (High Fare)
HF = df["Fare"]>100
df[HF].shape[0]


# In[61]:


#Number of passengers with Fare>100 who survived
df[HF & Surv].shape[0]


# In[70]:


#Number of female passengers with Fare>100
df[Fem & HF].shape[0]


# In[66]:


#Number of female passengers with Fare>100 who survived
df[HF & Surv & Fem].shape[0]


# In[55]:


#Number of passengers with Fare<50 (Low Fare)
LF = df["Fare"]<50
df[LF].shape[0]


# In[62]:


#Number of passengers with Fare<50 who survived
df[LF & Surv].shape[0]


# In[64]:


#Survival rate (in %), among female passengers
FemSurv = df[Fem & Surv].shape[0]/df[Fem].shape[0] * 100
FemSurv


# In[65]:


#Survival rate (in %), among passengers with high fare
HFSurv = df[HF & Surv].shape[0]/df[HF].shape[0] * 100
HFSurv


# In[69]:


#Survival rate (as a % of ALL Female Passengers), among female passengers with high fare
FemHFSurv = df[HF & Surv & Fem].shape[0]/df[Fem & HF].shape[0] * 100
FemHFSurv


# In[72]:


#Survival rate (in %), among passengers with low fare
LFSurv = df[LF & Surv].shape[0]/df[LF].shape[0] * 100
LFSurv

