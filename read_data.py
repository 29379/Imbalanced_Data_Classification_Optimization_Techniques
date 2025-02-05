import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



def read_data():
    print('Reading data...\n')
    df = pd.read_csv('dataset/creditcard.csv')
    df = df.drop(['Time'], axis=1)
    
    temp = df['Amount'].values.reshape(-1, 1)
    df['Amount'] = StandardScaler().fit_transform(temp)
    
    count_non_fraud = df[df['Class'] == 0]
    count_fraud = df[df['Class'] == 1]
    count_other = df[(df['Class'] != 1) & (df['Class'] != 0)]
    IR = count_non_fraud.shape[0] / count_fraud.shape[0]
    print(f'Non-fraud: {count_non_fraud.shape[0]}\nFraud: {count_fraud.shape[0]}\nOther: {count_other.shape[0]}\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n')
    print(f'Imbalance Ratio: {IR:.3f}\n')

    dataset = pd.concat([count_non_fraud, count_fraud], axis=0)
    dataset = dataset.sample(frac=1)    # shuffle the dataset

    X = dataset.drop('Class', axis=1)
    y = dataset['Class']
    print(f'X shape: {X.shape}\ny shape: {y.shape}\n')
    # X = X[1:10000]
    # y = y[1:10000]

    return np.array(X), np.array(y)
