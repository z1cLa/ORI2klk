import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt

def calculate_rmse(predicted,true):
    return np.sqrt(((predicted-true)**2).mean())


def remove_outliers(df_in,col_name,scale=1.5):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1
    fence_low = q1 - scale*iqr
    fence_high = q3 + scale*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

df = pd.read_csv("CLF_REG/data/train.csv")

df = df.dropna()

df = pd.get_dummies(df,columns=['zvanje','oblast','pol'], drop_first=True)
df = df.drop('pol_Male', axis=1)

train,test = train_test_split(df,test_size=0.3,random_state=42)

train = remove_outliers(train,'godina_doktor')
train = remove_outliers(train,'godina_iskustva')

x_train = train.drop("plata",axis = 1)
y_train = train['plata']


st = StandardScaler()

st.fit(x_train)

x_train[x_train.columns] = st.transform(x_train[x_train.columns])

x_test = test.drop("plata", axis =1)
y_test = test['plata']

x_test[x_test.columns] = st.transform(x_test[x_test.columns])

mlp = MLPRegressor(hidden_layer_sizes=[50,50,20], learning_rate_init=0.01, max_iter=20000, verbose=True, random_state=42).fit(x_train,y_train)
y_pred = mlp.predict(x_test)


plt.plot(np.arange(len(mlp.loss_curve_)),mlp.loss_curve_)
plt.show()

# METRIKA
# y_pred -> predikcije vaseg modela
# y_test -> prave vrednosti iz csv
RMSE = calculate_rmse(y_pred, y_test)
print(f'RMSE: {RMSE}') 
