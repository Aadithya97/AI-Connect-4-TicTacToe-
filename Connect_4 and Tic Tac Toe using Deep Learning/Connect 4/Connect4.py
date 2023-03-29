#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import pickle
import copy


# In[2]:


#Setting the seed to main consistency across models for comparison
np.random.seed(5)


# In[3]:


#1 is player X, 0 is player O (second player) -1 is blank
def function_to_num(array):
    array=array.reshape(42,)
    #print(array)
    for i in range(len(array)):
        if array[i]=='b':
            array[i]=-1
        elif array[i]=='x':
            array[i]=1
        else:
            array[i]=0
    return array.tolist()


# In[4]:


def function_to_alphabet(array):
    array=np.array(array)
    array=array.reshape(42,)
    #print(array)
    for i in range(len(array)):
        if array[i]==-1:
            array[i]='b'
        elif array[i]==1:
            array[i]='x'
        else:
            array[i]='o'
    #return array.tolist()
    return np.array(array)


# In[5]:


filename = 'mlp_classifier_connect_4'
loaded_model = pickle.load(open(filename, 'rb'))
def bot_playing_game(state,model):
    state=np.array(state,dtype=object)
    state=function_to_num(state)
    state=np.array(state)
    possible_new_state=[]
    all_indices=[]
    for i in range(7):
        for step in range(6):
            index=(41-i)-(7*step)
            if state[index]==-1:
                all_indices.append(index)
                break
    for i in all_indices:
        if state[i]==-1:
            possible_new_state=copy.deepcopy(state)
            possible_new_state[i]=0
            pred_state=np.expand_dims(possible_new_state, axis=0)
            if model.predict(pred_state)==1:
                break
    possible_new_state=np.array(possible_new_state,dtype=object)
    return_state=np.array(function_to_alphabet(possible_new_state))
    return_state=return_state.reshape(6,7)
    return return_state


def main(state):
    print(bot_playing_game(state,loaded_model))
# In[ ]:




