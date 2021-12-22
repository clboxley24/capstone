# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 08:21:46 2021

@author: CLB168
"""

import pandas as pd
import os
import re
import numpy as np
import math
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import preprocessing
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

# read in the json files
portfolio = pd.read_json('C:/Users/CLB168/Documents/4_udacity/portfolio.json', orient='records', lines=True)
profile = pd.read_json('C:/Users/CLB168/Documents/4_udacity/profile.json', orient='records', lines=True)
transcript = pd.read_json('C:/Users/CLB168/Documents/4_udacity/transcript.json', orient='records', lines=True)

# temporary variable to store merged profile and transcript data
temp = profile.merge(transcript, how='inner', left_on='id', right_on='person')

# dataframe that filters for only offers received or viewed
df = temp[(temp['event'] == 'offer received') | (temp['event'] == 'offer viewed')]
df.reset_index(drop=True, inplace=True)

# create new column with offer IDs
offer = []
for i in range(0, len(df)):
    x = df['value'][i]['offer id']
    offer.append(x)
    
df['offer'] = offer


def transform_data(df):
    
    """
    Description: 
      Transforms given data by making each row a unique combination person and offer attempt
    
    Parameters:
        df (DataFrame) - Combined dataframe containing merged data from profile.json
                        and transcript.json
    
    Output:
        final_df (DataFrame) - Transformed dataframe
    """
    
    # sort and reset index
    df.sort_values(by=['id', 'time', 'event'], ascending=[True, True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # subset of columns
    df_subset = df[['person', 'event', 'offer', 'time']]
    
    offer_number = []
    
    #loop through and add offer number to person IDs
    count = 1
    for i in range(0,len(df_subset)):
        # if first row, add _1
        if i == 0:
            offer_number.append(str(df_subset['person'][i]) + '_' + str(count))
        else:
            # if new person, restart the count
            if df_subset['person'][i] != df_subset['person'][i-1]:
                count = 0
            else:
                pass
            # if offer was viewed, maintain same count; increase count by 1 for the next cycle
            if 'offer viewed' in df_subset['event'][i] and df_subset['offer'][i] != df_subset['offer'][i-1]:
                offer_number.append(str(df_subset['person'][i]) + '_' + str(count))
                count += 1
            else:
                # if new offer is received, increase count by 1
                if 'offer received' in df_subset['event'][i]:
                    count += 1
                    offer_number.append(str(df_subset['person'][i]) + '_' + str(count))
                    
                else:
                    offer_number.append(str(df_subset['person'][i]) + '_' + str(count))
                        
    df_subset['offer_number'] = offer_number
    
    print(df_subset['offer_number'].value_counts())

    # flatten -  every row is a unique offer number 
    cc = df_subset.groupby('offer_number').cumcount() + 1
    
    df_final = df_subset.set_index(['offer_number', 'person', 'offer', cc]).unstack().sort_index(1, level=1)

    df_final.columns = ['_'.join(map(str,i)) for i in df_final.columns]
    df_final.reset_index(inplace=True)
    
    
    # dataframe cleaning
    df_final = df_final[['offer_number', 'person', 'offer', 'time_1', 'time_2']]
    df_final.rename(columns={'time_1': 'time_received', 'time_2': 'time_viewed'}, inplace=True)
    
    df_final['viewed'] = np.where(pd.isna(df_final['time_viewed']), 0, 1)
    
    return(df_final)

df = transform_data(df)

# merge our final dataset with portfolio and re-merge with profile
df = df.merge(portfolio, how='left', left_on='offer', right_on='id')
df = df.merge(profile, how='left', left_on='person', right_on='id')
# drop duplicate columns
df.drop(columns=['id_x', 'id_y'], inplace=True)

# create new field for time difference
import datetime
df['c.became_member_on'] = pd.to_datetime(df['became_member_on'], format='%Y%m%d')

#loop through and calculate how long each person has been a member
time_difference = []
for i in range(0, len(df)):
    difference = abs(df['c.became_member_on'][i] - datetime.datetime.now()).days
    time_difference.append(difference)
        
df['time_difference'] = time_difference

# create new column for income groups
income_group = []
for income in df['income']:
    if income < 25000.0:
        income_group.append('25k or less')
    elif income > 25000.0 and income <= 50000.0:
        income_group.append('25k-50k')
    elif income > 50000.0 and income <= 75000.0:
        income_group.append('50k-75k')
    elif income > 75000.0 and income <= 100000.0:
        income_group.append('75k-100k')
    elif income > 100000.0 and income <= 125000.0:
            income_group.append('100k-125k') 
    elif income < 125000.0:
            income_group.append('125k<') 
    else:
        income_group.append('n/a')
    
df['income_group'] = income_group

# create new column for age groups
age_group = []
for age in df['age']:
    if age < 25:
        age_group.append('25 or younger')
    elif age > 25 and age <= 40:
        age_group.append('25-40')
    elif age > 40 and age <= 55:
        age_group.append('40-55')
    elif age > 55 and age <= 70:
        age_group.append('55-70')
    elif age > 75:
        age_group.append('75<') 
    else:
        age_group.append('n/a')
    
df['age_group'] = age_group

# bar chart showing view percentage by offer type
offer_type_percent = df.groupby('offer_type').apply(lambda x: x['viewed'].sum()/len(x)).to_frame().reset_index()
offer_type_percent.rename(columns={0: 'percentage'}, inplace=True)

import seaborn as sns
sns.set_theme(style='whitegrid')
ax = sns.barplot(x='offer_type', y='percentage', data=offer_type_percent)
ax.set(ylim=(0, 1.0))

for index, row in offer_type_percent.iterrows():
    ax.text(row.name,row.percentage, round(row.percentage,3), color='black', ha='center')
    
# bar chart showing view percentage by age group
age_percent = df.groupby('age_group').apply(lambda x: x['viewed'].sum()/len(x)).to_frame().reset_index()
age_percent.rename(columns={0: 'percentage'}, inplace=True)

import seaborn as sns
sns.set_theme(style='whitegrid')
ax = sns.barplot(x='age_group', y='percentage', data=age_percent)
ax.set(ylim=(0, 1.0))

# bar chart showing view percentage by income group
income_percent = df.groupby('income_group').apply(lambda x: x['viewed'].sum()/len(x)).to_frame().reset_index()
income_percent.rename(columns={0: 'percentage'}, inplace=True)

import seaborn as sns
sns.set_theme(style='whitegrid')
ax = sns.barplot(x='income_group', y='percentage', data=income_percent, order=['25k-50k','50k-75k','75k-100k','100k-125k'])
ax.set(ylim=(0, 1.0))

##########
# logistic regression model to predict who will view offer based on age, income, and days since joining

# drop rows with NaNs
data = df.dropna(subset=['age', 'income', 'time_difference'])

#create dataframe with the following fields
data = data[['age'
           ,'income'
           ,'time_difference'
           ,'viewed']] # extra predictor variables we can add later - 'Total Encounters (group)','days_since_last_appt_calc (group)',

X = data.loc[:, data.columns != 'viewed'] # create dataframe of predictor variables 
y = data.loc[:, data.columns == 'viewed'] # create dataframe of outcome variable(s)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # split up data into training and testing subsets

model = LogisticRegression(class_weight={1:1, 0:3}) # create model object
model.fit(X_train,y_train) # fit model to training data
 
#----------------------
   
def test_model(subset):
    """
    Description: 
      Tests logistic regression model on training or test data
    
    Parameters:
        subset (string) - can either be 'train' or 'test' depending on what subset
                          of the data the user wants to test
    
    Output:
        conf_matrix (array) - Confusion matrix that details true positives, false positives, 
                             true negatives and false positives
    """
    
    if subset.lower() == 'train':
    
        y_pred=model.predict(X_train) # create a list of predictions based on predictor variables
    
        y_true = y_train['viewed']
        y_true = y_true.to_numpy()
        
        conf_matrix = confusion_matrix(y_true, y_pred) #, rownames=['Actual'], colnames=['Predicted'])
    
    else:

        y_pred=model.predict(X_test) # create a list of predictions based on predictor variables
        
        y_true = y_test['viewed']
        y_true = y_true.to_numpy()
        
        conf_matrix = confusion_matrix(y_true, y_pred) #, rownames=['Actual'], colnames=['Predicted'])
        
    
    # Accuracy
    from sklearn.metrics import accuracy_score
    print('accuracy = '+ str(accuracy_score(y_true, y_pred)))
    print('\n')
    
    # Recall
    from sklearn.metrics import recall_score
    print('recall = '+ str(recall_score(y_true, y_pred)))
    print('\n')
    
    # Precision
    from sklearn.metrics import precision_score
    print('precision = '+ str(precision_score(y_true, y_pred)))
    print('\n')
    
    from sklearn.metrics import f1_score
    print('f1 = '+ str(f1_score(y_true, y_pred)))
    
    print('\n')
    print('Classification report:')
    print(classification_report(y_true, y_pred))
    print('\n')
    print('Model coefficients:')
    # print(model.coef_)
    # print(X.columns)
    
    temp = model.coef_.tolist()
    coefficient_list = [item for sublist in temp for item in sublist]
    dictionary = {'Variable':X.columns, 'Coefficient':coefficient_list}
    coefficients = pd.DataFrame(dictionary)
    print(coefficients)
    
    return(conf_matrix)

lr_conf_matrix = test_model('test')

#########
# random forest classifier to predict who will view offer based on age, income, and days since joining

from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               max_features='auto',
                               bootstrap=True,
                               random_state=42,
                               class_weight={1:1, 0:1})

# Fit on training data
model.fit(X_train, np.ravel(y_train))

#----------------------
   
def test_model(subset):
    """
    Description: 
      Tests random forest model on training or test data
    
    Parameters:
        subset (string) - can either be 'train' or 'test' depending on what subset
                          of the data the user wants to test
    
    Output:
        conf_matrix (array) - Confusion matrix that details true positives, false positives, 
                             true negatives and false positives
    """
    
    if subset.lower() == 'train':
    
        y_pred=model.predict(X_train) # create a list of predictions based on predictor variables
    
        y_true = y_train['viewed']
        y_true = y_true.to_numpy()
        
        conf_matrix = confusion_matrix(y_true, y_pred) #, rownames=['Actual'], colnames=['Predicted'])
    
    else:

        y_pred=model.predict(X_test) # create a list of predictions based on predictor variables
        
        y_true = y_test['viewed']
        y_true = y_true.to_numpy()
        
        conf_matrix = confusion_matrix(y_true, y_pred) #, rownames=['Actual'], colnames=['Predicted'])
    
    print('\n')
    print('Classification report:')
    print(classification_report(y_true, y_pred))
    print('#---------------')
    
    if subset.lower() == 'test':
        
    
        y_pred=model.predict(X_test) # create a list of predictions based on predictor variables
    
        y_true = y_test['viewed']
        y_true = y_true.to_numpy()
        
        conf_matrix = confusion_matrix(y_true, y_pred) #, rownames=['Actual'], colnames=['Predicted'])
    
    else:

        y_pred=model.predict(X_test) # create a list of predictions based on predictor variables
        
        y_true = y_test['viewed']
        y_true = y_true.to_numpy()
        
        conf_matrix = confusion_matrix(y_true, y_pred) #, rownames=['Actual'], colnames=['Predicted'])
    
    print('\n')
    print('Classification report:')
    print(classification_report(y_true, y_pred))
    print('#---------------')
    
    return(conf_matrix)


rf_conf_matrix = test_model('test')
