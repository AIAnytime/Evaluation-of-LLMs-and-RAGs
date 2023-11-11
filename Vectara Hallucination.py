#!/usr/bin/env python
# coding: utf-8

# ## Cross-Encoder for Hallucination Detection
# 
# This model was trained using **SentenceTransformers Cross-Encoder class**. The model outputs a probabilitity from 0 to 1, **0 being a hallucination and 1 being factually consistent**. The predictions can be thresholded at 0.5 to predict whether a document is consistent with its source.

# #### The model can be used like below, on pairs of documents, passed as a list of list of strings (List[List[str]]]):

# In[1]:


from sentence_transformers import CrossEncoder


# In[2]:


model = CrossEncoder('vectara/hallucination_evaluation_model')


# In[3]:


model 


# In[5]:


scores = model.predict([
    ["A man walks into a bar and buys a drink", "A bloke swigs alcohol at a pub"],
    ["A person on a horse jumps over a broken down airplane.", "A person is at a diner, ordering an omelette."],
    ["A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."],
    ["A boy is jumping on skateboard in the middle of a red bridge.", "The boy skates down the sidewalk on a blue bridge"],
    ["A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond drinking water in public."],
    ["A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond man wearing a brown shirt is reading a book."],
    ["Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."],  
])


# In[6]:


scores


# In[9]:


import numpy as np 


# In[10]:


# Convert the values to one decimal point
score_one_decimal = np.around(scores, decimals=1)

# Convert the values to percentage with one decimal point
score_percentage = np.around(scores * 100, decimals=1)


# In[11]:


score_one_decimal


# In[12]:


score_percentage


# In[ ]:




