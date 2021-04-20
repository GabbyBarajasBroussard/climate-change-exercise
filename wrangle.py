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

# In[3]:


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
    #Drop city, state, latitude and longitude since the data has been filtered for houston
    df.drop(columns={'State', 'City', 'Latitude', 'Longitude'}, inplace=True)
    return df
# In[ ]:
def numeric_hists(df, bins=20):
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




