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

###################### Acquire Titanic Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def new_titanic_data():
    '''
    This function reads the titanic data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM passengers'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    
    return df



def get_titanic_data():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('titanic_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('titanic_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_titanic_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('titanic_df.csv')
        
    return df

def impute_age (train, validate, test):
    '''imputes the mean age of train to all data sets'''
    imputer = SimpleImputer(strategy = "mean", missing_values = np.nan)
    imputer = imputer.fit(train[["age"]])
    train[["age"]] = imputer.transform(train[["age"]])
    validate[["age"]] = imputer.transform(validate[["age"]])
    test[["age"]] = imputer.transform(test[["age"]])
    return train, validate,test

def clean_titanic_data(df):
    '''Takes in a Titanic DF and returns a clean DF 
       arguments - a pandas df with the expected feature names and columns 
       Return:clea_df a df with cleaning operation perfomed on it '''
    df.drop_duplicates(inplace = True)
    columns_to_drop = ['embarked','class','passenger_id','deck']
    df = df.drop(columns = columns_to_drop)
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na =False, drop_first = [True,True])
    df = pd.concat([df, dummy_df], axis = 1)
    return df.drop(columns=["sex", "embark_town"])

def prep_titanic_data(df):
    df= clean_titanic_data(df)
    train, test =train_test_split(df,
                                 train_size = 0.8,
                                 stratify = df.survived,
                                 random_state = 1234)
    train, validate = train_test_split(train,
                                      train_size = 0.7,
                                      stratify = train.survived,
                                      random_state = 1234)
    train, validate, test = impute_age(train, validate, test)
    return train, validate, test



###################### Acquire Iris Data ######################

def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT 
                    species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                FROM measurements
                JOIN species USING(species_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    
    return df


def get_iris_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('iris_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('iris_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_iris_data()
        
        # Cache data
        df.to_csv('iris_df.csv')
        
    return df

# def prep_iris(df_iris):
#     columns_to_drop = ["species_id"]
#     df_iris = df_iris.drop(columns = columns_to_drop)
#     df_iris = df_iris.rename(columns = {"species_name": "species"})
#     df_iris_new_dummies = pd.get_dummies(df_iris[["species"]])
#     df_iris = pd.concat([df_iris_new_dummies,df_iris], axis =1)
#     return df_iris
def prep_iris(df_iris):
    columns_to_drop = ["species_id"]
    df_iris = df_iris.drop(columns = columns_to_drop)
    df_iris = df_iris.rename(columns = {"species_name": "species"})
    df_iris_new_dummies = pd.get_dummies(df_iris[["species"]])
    df_iris = df_iris.drop(columns  =  "species")
    df_iris = pd.concat([df_iris_new_dummies,df_iris], axis =1)
    return df_iris
################ Telco Data ####################3
    
def new_telco_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df

def get_telco_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df


def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn_Yes)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn_Yes)
    return train, validate, test

def prep_telco_data(df):
    '''Takes in a telco DF and returns a clean DF 
       arguments - a pandas df with the expected feature names and columns 
       Return:clea_df a df with cleaning operation perfomed on it '''
    columns_to_drop= ['payment_type_id', 'internet_service_type_id','contract_type_id','customer_id','phone_service']
    df = df.drop(columns = columns_to_drop)
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges !=""] #(here we decided todrop the rows with missing values)
    df['total_charges']= pd.to_numeric(df['total_charges'], downcast='float')
    df.replace("No internet service", "No", inplace= True)
    df.replace("No phone service", "No", inplace= True) #getting rid of redundant options in the no phone service and no internet service columns 
    telco_dummy_df = pd.get_dummies(df[['gender', 'partner','dependents','multiple_lines','online_security','online_backup','device_protection','tech_support','churn','streaming_tv','streaming_movies','paperless_billing','contract_type','internet_service_type','payment_type']], dummy_na =False, drop_first = True)
    df = pd.concat([df, telco_dummy_df], axis =1)
    columns_dropped = ['gender', 'partner','dependents','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','churn','streaming_movies','paperless_billing','contract_type','internet_service_type','payment_type']
    df = df.drop(columns = columns_dropped)
    df.columns = df.columns.str.replace(' ', '_')

       # split the data
    # return df
    train, validate, test = split_telco_data(df)
    
    return train, validate, test

    #################### CONFUSION MATRIX CHART #########


def show_scores(TN, FP, FN, TP):
    
    ALL = TP + TN + FP + FN
    
    accuracy = (TP + TN)/ALL # How often did the model get it right?
    precision = TP/(TP+FP) # What is the quality of a positive prediction made by the model?
    recall = TP/(TP+FN) # How many of the true positives were found?   
    
    true_positive_rate = TP/(TP+FN) # Same as recall, actually
    true_negative_rate = TN/(TN+FP) # How many of the true negatives were found?
    false_positive_rate = FP/(FP+TN) # How often did we miss the negative and accidentally call it positive?
    false_negative_rate = FN/(FN+TP) # How often did we miss the positive and accidentally call it negative?
    
    f1_score = 2*(precision*recall)/(precision+recall) # Harmonic mean, good for imbalanced data sets
    support_pos = TP + FN # Number of actual positives in the sample
    support_neg = FP + TN # Number of actual negatives in the sample
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"True Positive Rate: {true_positive_rate}")
    print(f"True Negative Rate: {true_negative_rate}")
    print(f"False Positive Rate: {false_positive_rate}")
    print(f"False Negative Rate: {false_negative_rate}")
    print(f"F1 Score: {f1_score}")
    print(f"Support (0): {support_pos}")
    print(f"Support (1): {support_neg}")

# functions for tenure plot
def tenure_vs_churn(telco_df):
    tenure_not_churn = telco_df[telco_df.churn == "No"].tenure
    tenure_churn = telco_df[telco_df.churn == "Yes"].tenure


    plt.ylabel("$ customers $")
    plt.xlabel ("$ tenure $")
    plt.title ("Tenure and Churned Customers")
    plt.hist([tenure_churn, tenure_not_churn], 
              color = ["red", "blue"], 
              label = ["churn_yes", "churn_no"]) 
    plt.legend()
    return 

# functions for monthly_charges_churn plot

def monthly_charges_churn(telco_df):  
    mc_not_churn = telco_df[telco_df.churn == "No"].monthly_charges
    mc_churn = telco_df[telco_df.churn == "Yes"].monthly_charges


    plt.ylabel("$ customers $")
    plt.xlabel ("$ monthly charge $")
    plt.title ("Tenure and Churned Customers")
    plt.hist([mc_churn, mc_not_churn], 
          color = ["red", "blue"], 
          label = ["churn_yes", "churn_no"]) 
    plt.legend()
    return



# def prep_telco_data(df):
#     # Drop duplicate columns
#     df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
       
#     # Drop null values stored as whitespace    
#     df['total_charges'] = df['total_charges'].str.strip()
#     df = df[df.total_charges != '']
    
#     # Convert to correct datatype
#     data['total_charges']= pd.to_numeric(data['total_charges'], downcast='float')
    
#     # Convert binary categorical variables to numeric
#     df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
#     df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
#     df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
#     df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
#     df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
#     df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
#     # Get dummies for non-binary categorical variables
#     dummy_df = pd.get_dummies(df[['multiple_lines', \
#                               'online_security', \
#                               'online_backup', \
#                               'device_protection', \
#                               'tech_support', \
#                               'streaming_tv', \
#                               'streaming_movies', \
#                               'contract_type', \
#                               'internet_service_type', \
#                               'payment_type']], dummy_na=False, \
#                               drop_first=True)
    
#     # Concatenate dummy dataframe to original 
#     df = pd.concat([df, dummy_df], axis=1)
#     return df