#!/usr/bin/env python
# coding: utf-8

# # TASK 4- EMAIL SPAM DETECTION WITH MACHINE LEARNING

# In[132]:


# importing necessary libraries

# for numerical operations
import pandas as pd
import numpy as np

# for graphical visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[133]:


sms_df=pd.read_csv('C:/Users/preet/OneDrive/Desktop/spam.csv' , encoding = "ISO-8859-1")


# In[134]:


# note :- Column names in sms_df doesn't have any meaning
# thus, we will handle it in upcoming steps
sms_df.columns


# In[135]:


# (rows,columns)
sms_df.shape


# In[136]:


# rows*columns
sms_df.size


# In[137]:


# first 5 records
sms_df.head()


# In[138]:


# last five records
sms_df.tail()


# In[139]:


# random 5 records
sms_df.sample(5)


# In[140]:


sms_df.info()


# In[141]:


sms_df.isna().sum()


# NOTE :- Unnamed: 2,3,4 columns have only 50,12,6 not null values <br>
# thus, need to remove these columns , because we have 5572 records <br>
# and they have more than 5500 null values in it

# In[142]:


'''
axis=1 mean's dropping operation on columns
axis=0 mean's dropping operations on rows/records ( instead of names, it uses index id)

'''
sms_df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)


# In[143]:


# after dropping

sms_df.head()


# 3.2) Renaming the column names. <br>
# 'v1'  as 'target' <br>
# 'v2'  as 'sms'

# In[144]:


sms_df.rename(columns={'v1':'target','v2':'sms'},inplace=True)


# In[145]:


sms_df.head()


# 3.3 )  Check for duplicated values ( records ) in our dataframe

# In[146]:


# note :- duplicated() function returns boolean values 
#  False - >  duplicated record not found
#  True  - > duplicated record found

sms_df[sms_df.duplicated()]


# In[147]:


# total number of duplicated records in our dataframe
sms_df.duplicated().sum()


# In[148]:


# before removing duplicate records
sms_df.shape


# In[149]:


sms_df.drop_duplicates(inplace=True)


# In[150]:


# after removing duplicate records
sms_df.shape


# 
# Analysing Data using Visualisation plots (graphs )

# In[151]:


sms_df.columns


# In[152]:


# 'target' column contain higher 'ham' sms,
# thus, data seems unbalance


# Total number of 'ham' and 'spam' messages in 'target' column
sms_df['target'].value_counts()


# In[153]:


# in percentage
sms_df['target'].value_counts(normalize=True)*100


# #### Count plot
# Show the counts of observations in each categorical bin using bars. 

# In[154]:


# data is unbalanced

sns.countplot(x=sms_df['target'])
plt.show()


# #### Pie chart
# Pie charts can be used to show percentages of a whole,<br>
# and represents percentages at a set point in time. 

# In[155]:


# Calculating individal % of each category of 'Species' column

plt.pie(x=sms_df['target'].value_counts(),autopct='%.2f')
plt.title('Spam vs Ham')
plt.show()


# # 5) Feature Encoding
# converting text data into numeric form

# In[156]:


# converting 'spam' as 1 and 'ham' as 0 numeric value
sms_df['target']=sms_df['target'].map({'spam':1,'ham':0})


# In[157]:


# after changes
sms_df['target'].unique()


# In[158]:


sms_df.head()


# In[99]:


# nltk -> natural language tool kit
# PUNKT is an unsupervised trainable model tokenizer
# It tokenizer divides a text into a list of sentences by using an unsupervised algorithm
import nltk
nltk.download('punkt')


# In[100]:


import string

from nltk.corpus import stopwords
nltk.download('stopwords')


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[159]:


print(string.punctuation,"\n\n")


# In[160]:


print(stopwords.words('english'))


# In[161]:


def transform_sms(message):
    
    # to convert all characters in lower case
    message=message.lower()
    
    # to break sms record into words
    message=nltk.word_tokenize(message)
    
    # to remove special symbals
    temp=[]
    for i in message:
        if i.isalnum():
            temp.append(i)

    # creating clone of temp
    message=temp[:]   
    
    # clear the temp object
    temp.clear()
    
    # removing stopwords and punctuations
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            temp.append(i)
    
    message=temp[:]
    temp.clear()
    
    # stemming
    for i in message:
        temp.append(ps.stem(i))
    
    
    return " ".join(temp)


# In[162]:


# calling function 'transform_sms' by passing 'sms' records

sms_df['sms']=sms_df['sms'].apply(transform_sms)


# In[163]:


# after transformation
sms_df.head()


# #### Top 10 most used words in spam sms

# In[164]:


# for storing most used words
most_used_spam_words=[]

# .tolist() -> to convert 'series' object into 'list'
spam_list=sms_df[sms_df['target']==1]['sms'].tolist()

# accessing each individual elements from spam_list
for sentense in spam_list:
    
    # accessing each individual word form list elements
    for word in sentense.split():
        most_used_spam_words.append(word)


# In[165]:


# used for finding most comman words
from collections import Counter


# In[166]:


top_10=pd.DataFrame(Counter(most_used_spam_words).most_common(10))


# In[167]:


sns.barplot(x=top_10[0],y=top_10[1])
plt.show()


# 2) on ham records

# #### Top 10 most used words in spam sms

# In[168]:


# for storing most used words
most_used_ham_words=[]

# .tolist() -> to convert 'series' object into 'list'
ham_list=sms_df[sms_df['target']==0]['sms'].tolist()

# accessing each individual elements from spam_list
for sentense in ham_list:
    
    # accessing each individual word form list elements
    for word in sentense.split():
        most_used_ham_words.append(word)


# In[169]:


top_10_ham=pd.DataFrame(Counter(most_used_ham_words).most_common(10))


# In[170]:


sns.barplot(x=top_10_ham[0],y=top_10_ham[1])
plt.show()


# In[171]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[172]:


tfidf=TfidfVectorizer()


# In[173]:


# converting into vectors
x=tfidf.fit_transform(sms_df['sms']).toarray()


# In[174]:


x.shape


# In[175]:


x


# In[176]:


y=sms_df['target'].values


# In[177]:


y


# In[178]:


y.shape


# In[179]:


from sklearn.model_selection import train_test_split


# In[180]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[181]:


print("x_train - >  ",x_train.shape)
print("x_test - >  ",x_test.shape)
print("y_train - >  ",y_train.shape)
print("y_test - >  ",y_test.shape)


# ####  LogisticRegression :-

# In[182]:


from sklearn.linear_model import LogisticRegression


# In[183]:


model_lr=LogisticRegression()


# In[184]:


# train the model
model_lr.fit(x_train,y_train)


# In[185]:


# testing
y_pred_lr=model_lr.predict(x_test)
y_pred_lr


# In[186]:


y_test


# In[187]:


from sklearn.metrics import accuracy_score,precision_score


# In[188]:


print("accuracy score :- ",accuracy_score(y_test,y_pred_lr))
print("precision score :- ",precision_score(y_test,y_pred_lr))


# In[ ]:




