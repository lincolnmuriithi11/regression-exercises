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
    ''' this function calls a sql file from the codeup database and creates a data frame.
      '''
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        sql_query = pd.read_sql( '''
            SELECT 
            bedroomcnt,
            bathroomcnt,
            calculatedfinishedsquarefeet,
            taxvaluedollarcnt,
            yearbuilt,
            taxamount,
            fips,
            assessmentyear,
            propertyzoningdesc
            FROM
            properties_2017
                LEFT JOIN
            predictions_2017 USING (parcelid)
                JOIN
            propertylandusetype USING (propertylandusetypeid)
            WHERE
            propertylandusedesc = 'Single Family Residential'
            and
            column_name 
                AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
                AND bathroomcnt >= 1 <= 6
                AND bedroomcnt >= 1 <= 6
            '''
         df = pd.read_sql(sql_query, get_connection("zillow_df"))

        return df
        
df = get_zillow_sql()

# outlier handling to remove quant_cols with >3.5 z-score (std dev)
def remove_outliers(threshold, quant_cols, df):
    z = np.abs((stats.zscore(df[quant_cols])))
    df_without_outliers=  df[(z < threshold).all(axis=1)]
    print(df.shape)
    print(df_without_outliers.shape)
    return df_without_outliers
        

def wrangle_zillow(df):
    df = get_zillow_sql()
    df.dropna(axis=0, inplace=True)
    df = df[df.calculatedfinishedsquarefeet <= 10000]
    df = df[df.calculatedfinishedsquarefeet>70]
    df = df[df.bathroomcnt 1 <= 6]
    df = df[df.bedroomcnt 1 <= 6]
    df = df[df.taxvaluedollarcnt<=1_200_000]
    df["fips"] = pd.Categorical(df.fips)
    df['fips'].replace("6111.0",'Ventura', inplace=True)
    df['fips'].replace("6059.0",'Orange', inplace=True)
    df['fips'].replace("6037.0",'Los_Angeles', inplace=True)
    return df
# convert fips to catgorical
df = clean_df

# get dummies on fips
    

