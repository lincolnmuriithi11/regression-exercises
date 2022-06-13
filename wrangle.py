import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire and Clean Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
# ----------------------------------------------------------------- #
def get_zillow_sql():

    ''' this function calls a sql file from the codeup database and creates a data frame from the zillow db.
    '''
    query ='''
        SELECT 
        bedroomcnt,
        bathroomcnt,
        calculatedfinishedsquarefeet,
        taxvaluedollarcnt,
        yearbuilt,
        taxamount,
        fips
        FROM
        properties_2017
            LEFT JOIN
        predictions_2017 USING (parcelid)
        join
            propertylandusetype USING (propertylandusetypeid)
            WHERE propertylandusedesc = 'Single Family Residential'
            AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
            AND bathroomcnt >= 1 <= 6
            AND bedroomcnt >= 1 <= 6
        '''
    df = pd.read_sql(query, get_connection('zillow'))
    #creating a csv for easy access 
    return df
        
def get_zillow_df():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = get_zillow_sql()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df


# # outlier handling to remove quant_cols with >3.5 z-score (std dev)
# # this is another way of removing outliers
# def remove_outliers(df, calculated, columns):

#     z = np.abs((stats.zscore(df[quant_cols])))
#     df_without_outliers=  df[(z < threshold).all(axis=1)]
#     print(df.shape)
#     print(df_without_outliers.shape)
#     return df_without_outliers
        

def wrangle_zillow():
    """this function is for acquiring the zillow data, dropping nulls/ nan, removing outliers,converting the fips data to categorical, changing fips data to counties, renaming columns and saving the clean data in clean_zillow_data"""
    df = get_zillow_df()
    df.dropna(axis=0, inplace=True) # addressed missing values by dropping them since they were only less than 2% of the data
    df = df[df.calculatedfinishedsquarefeet <= 6000] #reoved outliers by cutting removing houses over 60000 sqfeet, below 70feet
    df = df[df.calculatedfinishedsquarefeet>70]
    df = df[df.taxvaluedollarcnt<=1_200_000]#removed houses over 1.2 million in dollar amount to remove outliers
    df = df[df.bedroomcnt<=6]# removed houses above 6 bedrooms/ 6baths  in order to have a normal distribution
    df = df[df.bathroomcnt<=6]
    df["fips"] = pd.Categorical(df.fips) #converted fips data to categorical and changed the names to county names for readability purposes
    df['fips'] = df['fips'].astype(str).apply(lambda x: x.replace('.0',''))
    df = df.rename(columns = {"bedroomcnt":"bedrooms", "bathroomcnt":"bathrooms","calculatedfinishedsquarefeet":"squarefeet","taxvaluedollarcnt": "total_taxes","fips":"county"})#renamed columns for easy readability as well
    df["county"].replace("6111",'Ventura', inplace=True)
    df["county"].replace("6059",'Orange', inplace=True)
    df["county"].replace("6037",'Los_Angeles', inplace=True)
    df.to_csv("clean_zillow.csv")# saved the wrangled data as a dataframe for easy access

    return df

# convert fips to catgorical

# get dummies on fips
    
# splitting dadta to train, validate, and test to avoid data leakage
def split_data(df):
    '''
    This function performs split on zillow data, stratify taxvaluedollarcnt.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123) 
                                            #  stratify=total_taxes) #in regression its not neccesary to scale your target variable
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=123) 
                                    #    stratify=train_validate.total_taxes)
    return train, validate, test



