import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Visualizamos la variable Embarked
'''
survived_embark = train_df[train_df['Survived']==1]['Embarked'].value_counts(sort=False)
dead_embark = train_df[train_df['Survived']==0]['Embarked'].value_counts(sort=False)
embark_df = pd.DataFrame([survived_embark,dead_embark],index=['Survived','Dead'])
embark_df.plot(kind='bar', stacked=True)
plt.xlabel('Embarked')
plt.ylabel('Number of passengers')
plt.title('Survival rate by port of embarkation')
st.pyplot()