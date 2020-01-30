
# coding: utf-8

# In[1]:


# Steven Jordan
# DSC 478 - Final Project
# Board Game Recommendation Engine

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing as skp


# In[2]:


#os.chdir('C:/Users/sajor/Documents/Career/Class/DSC 478/Final')


# In[3]:


# Reads the csv file, showing that there are 4999 entries, with 20 columns
df = pd.read_csv('bgg.csv')
print(df.shape)
print(df.head(3))


# In[4]:


# Initial Exploration, Data Cleaning, and Pre-processing

# Remove unnecessary columns
data_and_labels = df.drop(['rank', 'game_id', 'min_time', 'max_time', 'year', 'avg_rating', 'num_votes', 'image_url', 'owned', 'designer'], axis = 1)
print(data_and_labels.shape)

# Initial descriptions show there are non-unique names, and many games with missing values
#print(data_and_labels.describe(include = 'all'))


# In[5]:


# Remove games with missing numeric values (423 games)
# Most of these entries are simply missing from the website because they either weren't provided
# by the publisher, or they just don't appear to be encoded

data_and_labels = data_and_labels[data_and_labels.weight != 0]
data_and_labels = data_and_labels[data_and_labels.min_players != 0]
data_and_labels = data_and_labels[data_and_labels.max_players != 0]
data_and_labels = data_and_labels[data_and_labels.avg_time != 0]
data_and_labels = data_and_labels[data_and_labels.age != 0]
data_and_labels = data_and_labels[data_and_labels.category != 'none']
data_and_labels = data_and_labels[data_and_labels.mechanic != 'none']

data_and_labels.shape


# In[6]:


# Duplicate names appear to be different (though often related) games
#print(data_and_labels['names'].value_counts())


# In[7]:


# Explore non-categorical variables, detect and manage outliers
# Minimum number of players distribtion
plt.hist(data_and_labels['min_players'],  color = 'cyan', ec = 'black')
print(data_and_labels['min_players'].value_counts().sort_index())
plt.title('Distribution of Minimum PLayers\n')
plt.ylabel('Quantity')
plt.xlabel('Bin')

#IQR doesn't work for outlier detection, using domain knowledge
min_iqr = data_and_labels['min_players'].quantile(0.75) - data_and_labels['min_players'].quantile(0.25)
min_iqr_range = (data_and_labels['min_players'].quantile(0.25) - 1.5*min_iqr, data_and_labels['min_players'].quantile(0.75) + 1.5*min_iqr)
print("IQR: ", min_iqr)
print("1.5*IQR Range: ", min_iqr_range)


# In[8]:


# Domain knowledge, replacing all min_player above 4 values to 4. Accounts for 0.5% of games.
data_and_labels.loc[data_and_labels.min_players > 4, 'min_players'] = 4
plt.hist(data_and_labels['min_players'], bins = 4,  color = 'cyan', ec = 'black')
plt.title('Distribution of Minimum PLayers\n')
plt.ylabel('Quantity')
plt.xlabel('Bin')


# In[9]:


# Maximum number of players distribution
print(data_and_labels['max_players'].value_counts().sort_index())

#IQR makes sense, agrees with domain knowledge
max_iqr = data_and_labels['max_players'].quantile(0.75) - data_and_labels['max_players'].quantile(0.25)
max_iqr_range = (data_and_labels['max_players'].quantile(0.25) - 1.5*max_iqr, data_and_labels['max_players'].quantile(0.75) + 1.5*max_iqr)
print("IQR: ", max_iqr)
print("1.5*IQR Range: ", max_iqr_range)

plt.hist(data_and_labels['max_players'], bins = 40, color = 'cyan', ec = 'black')
plt.title('Distribution of Maximum PLayers\n')
plt.ylabel('Quantity')
plt.xlabel('Bin')


# In[10]:


# Domain knowledge, replacing all max_player above 9 values to 9. Accounts for 3.9% of games.
data_and_labels.loc[data_and_labels.max_players > 9, 'max_players'] = 9
print(data_and_labels['max_players'].value_counts().sort_index())
plt.hist(data_and_labels['max_players'], bins = 9, color = 'cyan', ec = 'black')
plt.title('Distribution of Maximum PLayers\n')
plt.ylabel('Quantity')
plt.xlabel('Bin')


# In[11]:


# Appears to be normally distributed
from statsmodels.graphics.gofplots import qqplot
qqplot(data_and_labels['max_players'], line='s')


# In[12]:


# Average game length distribution. Some games with crazy long or short average playtimes
print(data_and_labels['avg_time'].value_counts().sort_index())

# IQR
time_iqr = data_and_labels['avg_time'].quantile(0.75) - data_and_labels['avg_time'].quantile(0.25)
time_iqr_range = (data_and_labels['avg_time'].quantile(0.25) - 1.5*time_iqr, data_and_labels['avg_time'].quantile(0.75) + 1.5*time_iqr)
print("IQR: ", time_iqr)
print("1.5*IQR Range: ", time_iqr_range)

# Very skewed due to long play times
plt.hist(data_and_labels['avg_time'],bins = 100, color = 'cyan', ec = 'black')
plt.title('Distribution of Average Play Time\n')
plt.ylabel('Quantity')
plt.xlabel('Bin')


# In[13]:


# Replacing games' average time over 205 mins with 205 mins, accounts for 8.45% of games
data_and_labels.loc[data_and_labels.avg_time > 205, 'avg_time'] = 205
plt.hist(data_and_labels['avg_time'], bins = 9, color = 'cyan', ec = 'black')
plt.title('Distribution of Average Play Time\n')
plt.ylabel('Quantity')
plt.xlabel('Bin')


# In[14]:


# Minimum Age attribute
print(data_and_labels['age'].value_counts().sort_index())

# IQR is too restrictive - choosing not to implement it because it misses out on important types of games
age_iqr = data_and_labels['age'].quantile(0.75) - data_and_labels['age'].quantile(0.25)
age_iqr_range = (data_and_labels['age'].quantile(0.25) - 1.5*age_iqr, data_and_labels['age'].quantile(0.75) + 1.5*age_iqr)
print("IQR: ", age_iqr)
print("IQR Range: ", age_iqr_range)

# Appears to have a normal distribution with one outlier
plt.hist(data_and_labels['age'], color = 'cyan', ec = 'black')
plt.title('Distribution of Minimum Age\n')
plt.ylabel('Quantity')
plt.xlabel('Bin')


# In[15]:


# Remove the game with minimum age of 42 (it is null on the website)
data_and_labels = data_and_labels[data_and_labels.age != 42]
plt.hist(data_and_labels['age'], color = 'cyan', ec = 'black')
plt.title('Distribution of Minimum Age\n')
plt.ylabel('Quantity')
plt.xlabel('Bin')


# In[16]:


# Complexity distribution (weight) determined by BGG user ratings
plt.hist(data_and_labels['weight'], color = 'cyan', ec = 'black')
weight_iqr = data_and_labels['weight'].quantile(0.75) - data_and_labels['weight'].quantile(0.25)
weight_iqr_range = (data_and_labels['weight'].quantile(0.25) - 1.5*weight_iqr, data_and_labels['weight'].quantile(0.75) + 1.5*weight_iqr)
print("IQR: ", weight_iqr)
print("1.5*IQR Range: ", weight_iqr_range)

print('Min:', min(data_and_labels['weight']))
print('Max:',max(data_and_labels['weight']))

plt.hist(data_and_labels['weight'], color = 'cyan', ec = 'black')
plt.title('Distribution of Complexity Score\n')
plt.ylabel('Quantity')
plt.xlabel('Bin')


# In[17]:


# Adjust the higher end of the weight outliers to max IQR
data_and_labels.loc[data_and_labels.weight > 4.5945, 'weight'] = 4.5945
plt.hist(data_and_labels['weight'], color = 'cyan', ec = 'black')
plt.title('Distribution of Complexity Score\n')
plt.ylabel('Quantity')
plt.xlabel('Bin')


# In[18]:


# Create dummy variables for the mechanics (51 total different mechanics)
pd.options.display.max_columns = 100
mechanics_dummies = data_and_labels.mechanic.str.get_dummies(sep =', ')
mechanics_dummies.head(3)


# In[19]:


# The minimum # of mechanics is 1, the max is 18, and the mean is 3.02
mechs = []
for line in np.array(mechanics_dummies):
    mechs.append(sum(line))
    
print("Max: ", max(mechs))
print("Min: ", min(mechs))
print("Mean: ", np.mean(mechs))


# In[20]:


# Create dummy variables for the mechanics (83 total different mechanics)
category_dummies = data_and_labels.category.str.get_dummies(sep =', ')
category_dummies.head(3)


# In[21]:


# The minimum # of mechanics is 1, the max is 12, and the mean is 2.78
cats = []
for line in np.array(category_dummies):
    cats.append(sum(line))
    
print("Max: ", max(cats))
print("Min: ", min(cats))
print("Mean: ", np.mean(cats))


# In[22]:


# Remove old mechanic variables, add dummies
data_and_labels_dummies = data_and_labels.drop(['mechanic', 'category'], axis = 1)
data_and_labels_dummies = pd.concat([data_and_labels_dummies, mechanics_dummies, category_dummies], axis = 1)


# In[23]:


# Remove Expansions to base games
data_and_labels_dummies = data_and_labels_dummies[data_and_labels_dummies['Expansion for Base-game'] != 1]
data_and_labels_dummies = data_and_labels_dummies.drop(['Expansion for Base-game'], axis = 1)


# In[24]:


# Current size: 4575 different board games
# 3 Labels - URL, Names, geek_rating
# 5 numeric attributes - Min Players, Max Players, Avg Time, Min Age, Weight
# 51 mechanic dummy variables
# 82 category dummy variable
data_and_labels_dummies.head(3)


# In[25]:


# Separate out the data that will actually be used in the recommender system
data = data_and_labels_dummies.drop(['bgg_url', 'names','geek_rating'], axis = 1)
data.head(3)


# In[26]:


# Separate out the labels
labels = data_and_labels_dummies.loc[:,['bgg_url','names','geek_rating']]
labels.head(3)


# In[27]:


# Scale the data to reduce any individual variable's influence

import sklearn.preprocessing as skp

min_max_scaler = skp.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(data)
pd.DataFrame(data_scaled).head(3)

