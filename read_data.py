import pandas as pd


def read_data():
    df = pd.read_csv('data/creditcard.csv')
    
    count_non_fraud = df[df['Class'] == 0]
    count_fraud = df[df['Class'] == 1]
    count_other = df[(df['Class'] != 1) & (df['Class'] != 0)]
    print(f'Non-fraud: {count_non_fraud.shape[0]}\nFraud: {count_fraud.shape[0]}\nOther: {count_other.shape[0]}')
    
    dataset = pd.concat([count_non_fraud, count_fraud], axis=0)
    dataset = dataset.sample(frac=1)    # shuffle the dataset

    X = dataset.drop('Class', axis=1)
    y = dataset['Class']
    X = X[1:5000]
    y = y[1:5000]

    return X, y


if __name__ == "__main__":
    read_data()