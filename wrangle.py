#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp 
import os


import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('figure', figsize=(13, 9))
plt.rc('font', size=13)
from datetime import timedelta, datetime as dt

from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

############################################################################################################################################################################


def get_city_climate_data():
    '''This function reads in the csv, drops the unnamed column and returns as a dataframe.'''
    df= pd.read_csv('GlobalLandTemperaturesByCity.csv')
    return df

def get_country_climate_data():
    '''This function reads in the csv, drops the unnamed column and returns as a dataframe.'''
    df= pd.read_csv('GlobalLandTemperaturesByCountry.csv', index_col=0)
    return df
def get_major_city_climate_data():
    '''This function reads in the csv, drops the unnamed column and returns as a dataframe.'''
    df= pd.read_csv('GlobalLandTemperaturesByMajorCity.csv', index_col=0)
    return df
def get_state_climate_data():
    '''This function reads in the csv, drops the unnamed column and returns as a dataframe.'''
    df= pd.read_csv('GlobalLandTemperaturesByState.csv', index_col=0)
    return df
def get_global_temps_climate_data():
    '''This function reads in the csv, drops the unnamed column and returns as a dataframe.'''
    df= pd.read_csv('GlobalTemperatures.csv', index_col=0)
    return df

############################################################################################################################################################################
def prep_houston():
    '''This function aquires the data frame from the get_city_climate_data and then filters for the city of Houston in the USA. It then returns a dataframe of the climate data for Houston.'''
    #importing the city climate data
    og_df= get_city_climate_data()
    #making a new dataframe for the city of houston based on the original climate data
    df= og_df[((og_df['Country'] == 'United States') & (og_df["City"] == 'Houston'))]
    #Set dt to be datetime format
    df['dt'] = pd.to_datetime(df.dt, format='%Y-%m-%d')
    #set date time back to index
    df = df.set_index('dt').sort_index()
    #Drop city, latitude and longitude since the data has been filtered for houston
    df.drop(columns={'Country', 'City', 'Latitude', 'Longitude'}, inplace=True)
    return df
############################################################################################################################################################################
def numeric_hist_maker(df, bins=20):
    """
    Function to take in a DataFrame, bins default 20,
    select only numeric dtypes, and
    display histograms for each numeric column
    """
    plt.suptitle('Numeric Column Distributions')
    num_df = df.select_dtypes(include=np.number)
    num_df.hist(bins=bins, color='palevioletred', ec='mediumvioletred')
    plt.tight_layout()
    plt.show()
############################################################################################################################################################################
def split_houston_data():
    '''This function uses the prepped data and splits up the df into train, validate, test by using a percentage of the data. When using percentage based, use the last 20% as test'''
    df= prep_houston()
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df[: train_size]
    validate = df[train_size : validate_end_index]
    test = df[validate_end_index : ]
    
    return train, validate, test

###############
def make_predictions():
    avg_temp = train['AverageTemperature'][-1:][0]
    avg_temp_unc = train['AverageTemperatureUncertainty'][-1:][0]

    yhat_df = pd.DataFrame({'AverageTemperature': [avg_temp], 'AverageTemperatureUncertainty': [avg_temp_unc]}, 
                       index = validate.index)
    return yhat_df
###########################################################################################################################################################################    
def evaluate(target_var):
    '''evaluate() will compute the Mean Squared Error and the Rood Mean Squared Error to evaluate.'''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse
###########################################################################################################################################################################
def plot_and_eval(target_var):
    '''plot_and_eval() will use the evaluate function and also plot train and test values with the predicted values in order to compare performance.'''
    train, validate, test = split_houston_data()
    yhat_df
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()
    
########################################################################################################################################################################### 
    
# Create the empty dataframe
eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])

# function to store rmse for comparison purposes
def append_eval_df(model_type, target_var):
    rmse = evaluate(target_var)
    d = {'model_type': [model_type], 'target_var': [target_var], 'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)