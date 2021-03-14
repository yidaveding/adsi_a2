import pandas as pd
import numpy as np
import joblib
import os

def load_data(raw_filepath=None, cleaned_filepath=None):
    """
    loading data:
        if cleaned file path is provided and is also a valid path then load the processed file exist
        otherwise, load data and perform from raw data folder

    attributes:
        raw_filepath: String, raw data file path
        cleaned_filepath: String, cleaned data file path

    data cleansing method:
        keep only relevant columns, remove rows that contains NAs in the selected columns

    return:
        cleaned dataset
    """

    if (cleaned_filepath != None) & os.path.exists(cleaned_filepath):
        print('loading cleansed data')
        df_cleaned = pd.read_csv(cleaned_filepath)
    else:
        print('loading raw data')
        # select columns
        col_tar = ['beer_style']
        # col_cat = ['brewery_name']
        col_num = ['review_aroma', 'review_appearance', 'review_palate', 'review_taste']
        
        # load data from csv
        df = pd.read_csv(raw_filepath)

        # clean up dataset: drop unrelated columns and drop rows that contain NA
        df_cleaned = df.copy()
        df_cleaned = df[col_num + col_tar]
        df_cleaned.dropna(inplace=True)

        # store data
        df_cleaned.to_csv(cleaned_filepath, index=False)
    
    return df_cleaned

def scale_features(df, sc):
    """
    use provided scaler to scale cleansed data
    store fitted scaler into model folder

    input:
        df: pandas dataframe with numeric features
        sc: sklearn scaler

    output:
        save fitted scaler into model folder

    return:
        scaled data
    """

    df_scaled = sc.fit_transform(df)

    joblib.dump(sc, 'models/standard_scaler.joblib')

    return df_scaled

def encode_label(label, enc):
    """
    encode label with provided encoder

    input:
        label: pandas data series
        enc: encoder

    output:
        save fitted encoder into model folder

    return:
        encoded label
    """

    enc_target = enc.fit_transform(np.array(label))

    joblib.dump(enc, 'models/output_encoder.joblib')

    return enc_target