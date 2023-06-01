import pandas as pd
from sklearn.metrics import accuracy_score
import spacy # biblioteka za procesiranje teksta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('NB/data/south_park_train.csv')

df['Character'] = LabelEncoder().fit_transform(df['Character'])

df.dropna()

#x_train, x_test, y_train, y_test = train_test_split(df['Line'],df['Character'],test_size=0.2,random_state=496)
x_train, x_test, y_train, y_test = train_test_split(df,test_size=0.2,random_state=496)

vect = CountVectorizer(stop_words="english", ngram_range=((1,2)))
x_train = vect.fit_transform(x_train)
x_test = vect.transform(x_test)

print(f"Dimenzionalnost ulaznih podataka X {x_train.shape}")
nb = MultinomialNB().fit(x_train,y_train)
y_pred = nb.predict(x_test)

# METRIKA
accuracy = accuracy_score(y_pred, y_test)
print(f'Accuracy: {accuracy}')
