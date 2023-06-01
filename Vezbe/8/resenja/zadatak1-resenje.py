import pandas as pd
import numpy as np
from ann_comp_graph import *
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

def predict_mse(nn, test_x, test_y):
    total = 0
    for x, y in zip(test_x, test_y):
        res = nn.predict(x)
        total += (res-y)**2
        print(res, y)
    return total/len(test_x)

def normalize_rows(df, rows):
    for row in rows:
        df[row] = (df[row]-df[row].min())/(df[row].max()-df[row].min())
    return df

if __name__ == "__main__":
    df = pd.read_csv("life-expectancy.csv")
    df=df.sample(n=1000,random_state=200)
    df = df.drop('Country', axis=1)

    df = one_hot_encoding(df, 'Region')

    train, test = train_test_split(df, 0.8)


    train_y = np.array([train['Life_expectancy'].to_numpy()]).transpose()
    train_x = train.drop('Life_expectancy', axis=1).to_numpy()

    test_y = test['Life_expectancy'].to_numpy()
    test_x = test.drop('Life_expectancy', axis=1).to_numpy()

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    
    nn = NeuralNetwork()
    nn.add(NeuralLayer(train_x.shape[1], 16, 'sigmoid'))
    nn.add(NeuralLayer(16, 1, 'relu'))
    # nn.add(NeuralLayer(16, 1, 'relu'))

    
    history = nn.fit(train_x, train_y, learning_rate=0.1,  momentum=0.3, nb_epochs=10, shuffle=True, verbose=1)

    print(predict_mse(nn, test_x, test_y))