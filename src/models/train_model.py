import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(df, target, randomstate=42, path='data/processed/'):
    """
    split data into train, validation and test datasets
    store data as numpy array in data/processed folder
    """
    X_data, X_test, y_data, y_test = train_test_split(df, target, stratify=target, test_size=0.2,
                                                      random_state=randomstate)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, stratify=y_data, test_size=0.15, random_state=randomstate)

    if X_train is not None:
      np.save(f'{path}X_train', X_train)
    if X_val is not None:
      np.save(f'{path}X_val',   X_val)
    if X_test is not None:
      np.save(f'{path}X_test',  X_test)
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
    if y_val is not None:
      np.save(f'{path}y_val',   y_val)
    if y_test is not None:
      np.save(f'{path}y_test',  y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test