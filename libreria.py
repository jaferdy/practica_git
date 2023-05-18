import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Visualizamos la variable Sexo
'''
survived_sex = train_df[train_df['Survived']==1]['Sex'].value_counts(sort=False)
dead_sex = train_df[train_df['Survived']==0]['Sex'].value_counts(sort=False)
sex_df = pd.DataFrame([survived_sex,dead_sex],index=['Survived','Dead'])
sex_df.plot(kind='bar', stacked=True)
plt.xlabel('Sex')
plt.ylabel('Number of passengers')
plt.title('Survival rate by sex')
st.pyplot()