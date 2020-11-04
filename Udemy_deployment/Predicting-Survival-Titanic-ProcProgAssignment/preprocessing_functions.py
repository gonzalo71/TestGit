import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)



def divide_train_test(df, target):
   # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        df[target],
                                                        test_size=0.1,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test
    



def extract_cabin_letter(df, var):
    # captures the first letter
    # captures the first letter
    return df[var].str[0] 



def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    return np.where(df[var].isnull(), 1, 0)
    


    
def impute_na(df,var,replacement="Missing"):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    return df[var].fillna(replacement)



def remove_rare_labels(df,var,FREQUENT_LABELS,replacement="Rare"):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    return np.where(df[var].isin(
        FREQUENT_LABELS), df[var], replacement)



def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable     
        
    # to create the binary variables, we use get_dummies from pandas
    df = df.copy()
    
    df = pd.concat([df,
                    pd.get_dummies(df[var], prefix=var, drop_first=True)
                    	], axis=1)
    
    df.drop(labels=[var], axis=1, inplace=True)

    return df      
 
    
    



def check_dummy_variables(data, dummy_list):    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    data_col = data.columns.tolist()
    lista = [var for var in dummy_list if var not in data_col]
    if len(lista) == 0:
        print("all dummy variables added")
    else:
        for var in lista:
            data[var] = 0
    return data
    
    

def train_scaler(df, output_path):
    # train and save scaler
    # create scaler
    scaler = StandardScaler()
    #  fit  the scaler to the train set
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler
  
    

def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)
    return scaler.transform(df)



def train_model(df, target, output_path):
    # train and save model
    model = LogisticRegression(C=0.0005, random_state=0)
    # train the model
    model.fit(df, target)
     # save the model
    joblib.dump(model, output_path)
    
    return None



def predict(df, model):
    # load model and get predictions
    model = joblib.load(model)
    return model.predict(df)
