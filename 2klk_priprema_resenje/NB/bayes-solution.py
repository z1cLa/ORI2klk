import pandas as pd
import numpy as np
# from sklearn.base import accuracy_score
# import spacy # biblioteka za procesiranje teksta
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import re

df = pd.read_csv('NB/data/south_park_train.csv')
print(df)

"""
# 1. Enkodovanje labela.
"""
df['Character'] = LabelEncoder().fit_transform(df['Character'])

"""
# 2. Preprocesiranje teksta.
# Transformacija u mala slova.
    CountVectorizer već to radi, pa je redudantno to raditi ovde ukoliko se on koristi.
# Izbacivanje svih karaktera koji nisu alfanumerički karakteri ili whitespace karakteri.
    CountVectorizer već radi određeno preprocesiranje, pa je i ovo možda redundantno.
# Izbacivanje kratkih reči.
    Potrebno je biti oprezan sa izbacivanjem reči zbog toga što ponekad kratke reči nose bitnu informaciju za rešavanje nekih problema.
"""
# df['Line'] = df['Line'].apply(lambda x: x.lower())
# df['Line'] = df['Line'].apply(lambda x: re.sub(r"[^\w\s]","", x))
# df['Line'] = df['Line'].apply(lambda x: " ".join([word for word in x.split() if len(word) >= 2]))

"""
# 3. Izvacivanje redova u kojima fale vrednosti obeležja.
"""
df = df.dropna()

"""
4. Podela na trening i testni skup.
# Ukoliko se unapred odvoji kolona sa labelama od ostatka podataka, tada ova funkcija vraća i podatke i labelu za trening i testne skupove.
"""
x_train, x_test, y_train, y_test = train_test_split(df['Line'], df['Character'], test_size=0.2, random_state=496)


"""
5. Rad sa tekstom.
# CountVectorizer pravi vektorsku reprezentaciju teksta, 
  gde svaka vrednost u vektoru predstavlja broj pojavljivanja određene reči u rečenici.
# Hiperparametri za CountVectorizer.
    stop_words - Reči koje se često izbacuju iz rečenica prilikom rada sa tekstom.
        Ovde je stavljena kolekcija za engleski jezik.
    ngram_range - Torka u kojoj vrednosti predstavljaju minimalan i maksimalan broj reči koje će biti dodate u rečnik.
        Torka (1,2) znači da će se brojati pojedinačne reči, ali i svi parovi reči (bigrami).
# Fit se poziva samo na trening skupu. Ovde je pozvan fit_transform da se odma uradi i transformacija.
# Na testnom skupu se samo primeni transform funkcija.
"""
vect = CountVectorizer(stop_words="english", ngram_range=((1,2)))
x_train = vect.fit_transform(x_train)
x_test = vect.transform(x_test)

"""
# Bonus - TFIDF
# TfidfVectorizer (Term frequency inverse document frequency) se takođe može koristiti.
  U dugačkom tekstu bi CountVectorizer prebrojavo u proseku više reči nego u kratkom tekstu.
  Zbog toga se može izračunati Term Frequency (TF) tako što se broj pojavljivanja svake reči podeli sa brojem reči u tekstu.
  Inverse Document Frequency (IDF) označava koliko se često neka reč javlja u svim tekstovima. Što se češće reč javlja u tekstovima, IDF je manji.
  Reči koje se često javljaju u tekstovima nose manje informacija, pa se množenjem TF sa IDF umanjuje značaj čestih reči.
# Može se koristiti kao alternativa za CountVectorizer, jer neki modeli bolje rade sa Tfidf.
"""
# vect = TfidfVectorizer(stop_words='english', ngram_range=((1,1)))
# x_train = vect.fit_transform(x_train)
# x_test = vect.transform(x_test)

"""
6. Treniranje modela i vršenje predikcija.
# Multinomial Naive Bayes.
  Radi veoma brzo i postiže dobre performanse u klasifikaciji teksta.
  Može da radi brzo i kada se ngram_range poveća.
# MLPClassifier.
  Sporije se trenira, ali može ostvariti solidan rezultat.
  Prevelika dimenzionalnost ulaznih vektora ne odgovara neuronskoj mreži.
  CountVectorizer reprezentuje text vektorom koji ima onoliku dužinu koliko ima unikathin reči u skupu podataka.
  Zbog toga je bolje smanjiti ngram_range u vektorajzeru na (1,1), da bi se smanjila dimenzija vektora kada se radi sa neuronskom mrežom.

  Da bi se neuronska mreža efikasnije koristila, poželjno je drugačije reprezentovati tekst.
  U tu svrhu bi se mogli koristiti pretrenirani embedinzi fiksne dužine za reči kao što su Word2Vec, GloVe, ili FastText.
  To zahteva upotrebu dodatnih biblioteka, i dugo se izvrašava, pa nije ovde odrađeno.
# Dati su i primeri nekih drugih modela koji se mogu koristiti za klasifikaciju (SVC, SGDC).
"""
print(f"Dimenzionalnost ulaznih podataka X {x_train.shape}")
nb = MultinomialNB().fit(x_train, y_train)
y_pred = nb.predict(x_test)

# mlp = MLPClassifier(hidden_layer_sizes=[256, 128, 64], max_iter=5, learning_rate_init=0.01, verbose=True).fit(x_train, y_train)
# y_pred = mlp.predict(x_test)

# model = SVC().fit(x_train, y_train)
# y_pred = model.predict(x_test)

# model = SGDClassifier().fit(x_train, y_train)
# y_pred = model.predict(x_test)


# METRIKA
accuracy = accuracy_score(y_pred, y_test)
print(f'Accuracy: {accuracy}')
