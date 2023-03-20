#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st


# In[2]:


df = pd.read_csv("bankruptcy-prevention.csv", sep=";")
df


# In[3]:


df["class_as"] = 0

df.loc[df[" class"] == 'non-bankruptcy', 'class_as'] = 1
df.sample(10)


# In[4]:


df.drop(columns = ' class',axis =1,inplace = True)
df


# In[5]:


x = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[7]:


from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x,y=ros.fit_resample(x,y)


# In[8]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)


# In[9]:


clf_poly = SVC(kernel='poly')
clf_poly.fit(x_train , y_train)


# In[11]:


pickle_in = open("clf_poly.pkl","rb")
classifier=pickle.load(pickle_in)


# In[12]:


st.title("Bankruptcy Detector")


# In[13]:


def welcome():
    return "Welcome ALL"


# In[14]:


model_name = st.sidebar.selectbox(
    'Select Model',
    ('polySVC','DecisionTreeClassifier','BernoulisNaiveBayes','GradientBoostingClassifier')
)


# In[15]:


def predict_bankruptcy(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk):
    prediction= clf_poly.predict([[industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk]])
    print(prediction)
    return(prediction)


# In[16]:


def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bankruptcy Detector ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    industrial_risk = st.text_input("industrial_risk", "Type Here")
    management_risk = st.text_input("management_risk", "Type Here")
    financial_flexibility = st.text_input("financial_flexibility", "Type Here")
    credibility = st.text_input("credibility", "Type Here")
    competitiveness = st.text_input("competitiveness", "Type Here")
    operating_risk = st.text_input("operating_risk", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_bankruptcy(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk)
    st.success('The output is{}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")
if __name__=='__main__':
    main()


# In[ ]:




