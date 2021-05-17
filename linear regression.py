#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy import absolute, mean
# import libraries
df = pd.read_csv("train.csv")
df["FGApred"]="0"
df["Error"]="0"
df["PercentageError"]="0"
df["RsquaredU"]="0"
df["RsquaredT"]="0"
df
# read CSV file into variable "data" and display data + add in extra columns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("field goals made")
plt.ylabel("field goals attempted")
plt.scatter(df.FG,df.FGA,color="red")

# plot scatter graph of field goals made per game against field goals attempted per game from data


# In[3]:


x=np.array(df.FG)
y=np.array(df.FGA)
# assign FG and FGA to variables x and y


# In[4]:


linreg=LinearRegression() 
x=x.reshape(-1,1) #reshape data in variable x


# In[5]:


linreg.fit(x,y) #train the linear regression model with variables x and y 


# In[6]:


x=np.array(df.FG)
x=x.reshape(-1,1)
y=linreg.predict(x)
#predict y using regression model


# In[7]:


df.FGApred=y
df.Error=absolute(df.FGA-df.FGApred)
df.PercentageError=absolute((df.Error/df.FGA)*100)
df.RsquaredU=absolute(df.Error*df.Error)
average1=df["FGA"].mean()
df.RsquaredT=absolute((df.FGA-average1)*(df.FGA-average1))
df=df[np.isfinite(df).all(1)]
df
#calculate table values and display table for training data


# In[8]:


y_pred=linreg.predict(x)
plt.xlabel("field goals made")
plt.ylabel("field goal attempted")
plt.scatter(x,y,color="blue")
plt.scatter(df.FG,df.FGA,color="red")
plt.plot(x,y_pred,color="black")
plt.show() 
#plot graph of x,y/FG,FGA with linear regression model represented as straight line


# In[9]:


linreg.coef_
#calculate coeficient


# In[10]:


linreg.intercept_
#calculate intercept


# In[11]:


df["Error"].mean() # mean absolute error


# In[12]:


df["PercentageError"].mean() # mean absolute percentage error


# In[13]:


RsquaredUtotal1=df["RsquaredU"].sum()
RsquaredUtotal1


# In[14]:


RsquaredTotal1=df["RsquaredT"].sum()
RsquaredTotal1


# In[15]:


Rsquared1=1-(RsquaredUtotal1/RsquaredTotal1)
Rsquared1 #r squared value


# In[16]:


ESsum1=df["RsquaredU"].sum()
RMSE1=np.sqrt(ESsum1/982)
RMSE1 #root mean squared error


# In[17]:



test=pd.read_csv("test.csv")
test["FGApred"]="0" #FGA predicted values
test["Error"]="0" #absolute error
test["PercentageError"]="0" #absolute percentage error
test["RsquaredU"]="0" #R squared unexplained variance
test["RsquaredT"]="0" #R sqaured value total variance
test

#read in test data and add in needed columns


# In[18]:



x1=np.array(test.FG) #assign FG from test data to x1
x1=x1.reshape(-1,1) #reshape x1
y1=linreg.predict(x1) #predict FGA of the test data and assign to y1


# In[19]:


test.FGApred=y1 #assign predicted data to FGA of the test data
test.Error=absolute(test.FGA-test.FGApred)
test.PercentageError=absolute((test.Error/test.FGA)*100)
test.RsquaredU=absolute(test.Error*test.Error)
average=test["FGA"].mean()
test.RsquaredT=absolute((test.FGA-average)*(test.FGA-average))
#calculate absolute error, absolute percentage error, rsquared and root mean squared error
test=test[np.isfinite(test).all(1)]
#remove infinite values to avoid errors in calculations


# In[20]:


test


# In[21]:


test["Error"].mean()
#mean absolute error


# In[22]:


test["PercentageError"].mean()
#mean absolute percentage error


# In[23]:


RsquaredUtotal=test["RsquaredU"].sum()
RsquaredUtotal


# In[24]:


RsquaredTtotal=test["RsquaredT"].sum()
RsquaredTtotal


# In[25]:


Rsquared=1-(RsquaredUtotal/RsquaredTtotal)
Rsquared #calculate r squared value


# In[26]:


ESsum=test["RsquaredU"].sum()
RMSE=np.sqrt(ESsum/422)
RMSE #calculate root mean squared error


# In[27]:



plt.xlabel("field goals made")
plt.ylabel("field goal attempted")
plt.scatter(x1,y1,color="blue") #predicted values
plt.scatter(test.FG,test.FGA,color="red") #actual values
plt.show() 
#plot graph of test data with predicted FGA value


# In[28]:


test.to_csv("test results.csv")
df.to_csv("train results.csv")
#export data 


# In[ ]:




