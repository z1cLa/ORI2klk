import pandas as pd
import numpy as np
from ann_comp_graph import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def one_hot_encoding(df, column):
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = df.drop(column, axis=1)
    df = df.join(one_hot)

    return df

def train_test_split(df, percent):
    train=df.sample(frac=percent,random_state=200)
    test=df.drop(train.index)
    return train, test


def normalize_rows(df, rows):
    for row in rows:
        df[row] = (df[row]-df[row].min())/(df[row].max()-df[row].min())
    return df

if __name__ == "__main__":
    df = pd.read_csv("diabetes-prediction.csv")
    df=df.sample(n=3000,random_state=200)

    df = one_hot_encoding(df, 'gender')
    df = one_hot_encoding(df, 'smoking_history')

    train, test = train_test_split(df, 0.8)

    train_y = train['diabetes']
    train_x = train.drop('diabetes', axis=1)

    test_y = test['diabetes']
    test_x = test.drop('diabetes', axis=1)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    # za overfit primer koristiti n=3000 i max_iter=200 (daje bolji rezultat od max_iter=1000)
    clf = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=[256, 128], activation="relu", early_stopping=False, verbose=True).fit(train_x, train_y)
    print(clf.score(test_x, test_y))