import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

"""
# Za naprednije grafike može se koristiti seaborn biblioteka ukoliko je prethodno instalirana.
"""
# import seaborn as sns

def calculate_rmse(predicted, true):
    return np.sqrt(((predicted - true) ** 2).mean())

def remove_outliers(df_in, col_name, scale=1.5):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-scale*iqr
    fence_high = q3+scale*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

df = pd.read_csv("CLS_REG/data/train.csv")

"""
# 1. Na početku je poželjno analizirati podatke koji se nalaze u skupu podataka.
"""
print(df)

"""
# Ima mnogo više profesora u odnosu na druga zvanja.
"""
df["zvanje"].value_counts().plot.bar()
plt.show()

"""
# Podaci su izbalansirani po oblasti.
"""
df["oblast"].value_counts().plot.bar()
plt.show()

"""
# Podaci nisu balansirani kada je pol u pitanju, jer ima mnogo više muškaraca od žena u podacima.
"""
df["pol"].value_counts().plot.bar()
plt.show()


"""
# 2. Potrebno je rukovati nedostajućim vrednostima.
# Ukoliko postoje nedostajuće vrednosti u skupu podataka, jedna opcija je da se izbaci ceo red koji sadrži bar jednu nedostajuću vrednost.
  To se često radi ukoliko je mali broj redova u kojima nedostaju podaci.
# Alternativni pristup bi bio da se nedostajuće vrednosti popune srednom vrednošću za tu kolonu kod numeričkih obeležja, ili najčešćom kategorijom kod kategoričkih.
# Napredniji pristup je da se nađu najsličniji redovi kada se posmatraju preostale kolone koje imaju vrednosti,
  i da se zatim izračuna prosečna vrednost (najčešća kategorija) kolone k najsličnijih podataka i da se to stavi umesto nedostajuće vrednosti.
  U tu svrhu se može koristiti KNN algoritam.
"""
df = df.dropna()

"""
# 3. Kategorička obeležja je potrebno enkodovati.
# Jedna opcija je da se koristi LabelEncoder koji svakoj kategoriji dodeljuje ceo broj
  Problem sa LabelEncoder-om je što nekim kategorijama dodeljuje veći broj od drugih,
  pa na taj način definiše poredak koji možda ne bi trebao da postoji između kategorija.
# U slučaju labele za klasifikaciju, može se koristiti i label enkoder.
  Ovaj korak nije neophodan kod nekih modela, i MLPClassifier će raditi i ako se ne odradi ova transformacija.
"""
lenc = LabelEncoder()
df['zvanje'] = lenc.fit_transform(df['zvanje'])
# df['oblast'] = lenc.fit_transform(df['oblast'])
# df['pol'] = lenc.fit_transform(df['pol'])

"""
# Češće se koristi OneHotEncoding, gde je svaka kategorija predstavljena sa vektorom .
  koji sadrži sve nule i jedinicu na poziciji koja prestavlja tu kategoriju npr [0,1,0].
  U ovom slučaju je udaljenost svakog vektora koji predstavlja kategoriju od koordinatnog početka 1, pa nije definisan poredak.
  get_dummies metoda vrši OneHotEnkoding, i enkoduje n kategorija pomoću vektora veličine n-1,
  i dodaje svaku vrednost tog vektora kao posebnu kolonu. Zbog toga se odbace postojeće kolone sa drop_frist=True opcijom.
  Tako dobijene nove kolone imaju prefiks koji je oznaka stare kolone i na njega se dodaje ime kategorije npr. pol_Male.
"""
df = pd.get_dummies(df, columns=['oblast' ,'pol'], drop_first=True)

"""
# Dodatno - može se razmotriti izbacivanje kolone pol, zbog toga što su podaci u toj koloni veoma nebalansirani.
  Kod izbacivanja se uvek stavi axis=1 zbog toga što želimo da izbacimo kolonu.
"""
df = df.drop('pol_Male', axis=1)

"""
# 4. Skup podataka se treba podeliti na trening i testni skup.
# Može se napraviti 70%-30% podela za trening i testni skup.
# Ukoliko želimo da pri svakom pokretanju dobijemo istu podelu, možemo postaviti radnom_state i tako definisati seed za generator slučajnih brojeva.
"""
train, test = train_test_split(df, test_size=0.3, random_state=42)

"""
# 5. Dodatna analiza trening skupa podataka
# Korisno je proveriti da li postoje vrednosti koje značajno odstupaju od vrednosti ostalih podataka u istoj koloni. Takvi podaci se nazivaju "outliers".
# Za ovo se može koristi BoxPlot, gde kružići predstavljaju outlier-e.
"""
plt.boxplot(train['godina_iskustva'])
plt.show()

plt.boxplot(train['godina_doktor'])
plt.show()

"""
# Dodatno - Izvacivanje outlier-a
# Robustan način da se izbace outlier-i je da se koristi interkvartilni razmak IQR
  U tu svrhu je u ovaj zadatak ubačena posebna funkcija remove_outliers koja uklanja outlier-e u odabranoj koloni.
"""
train = remove_outliers(train, 'godina_doktor')
train = remove_outliers(train, 'godina_iskustva')


"""
# 6. Trening skup se zatim deli na kolone na kojima će model učiti (X) i na labele koje je potrebno predvideti (Y).
"""
x_train = train.drop("zvanje", axis=1)
y_train = train['zvanje']

"""
# 7. Da bi modeli mašinskog učenja bolje radili, poželjno je uraditi standardizaciju ili normalizaciju podataka.
# Standardizacija (još se zove i Z-Normalizacija) predstavlja proces gde se izračunaju srednja vrednost (mean) i standardna devijacija (std) za svaku kolonu
  i zatim se od svih vrednosti u toj koloni oduzme srednja vrednosti i podeli se sa standardnom devijacijom.
  Ovako se dobijaju male vrednosti centrirane oko nule, što u praksi pomaže modelima da bolje uče.
# Normalizacija predstavlja proces gde se sve vrednosti svedu na opseg od 0 do 1, i ređe se koristi od standardizacije. MinMaxScaler se koristi u tu svrhu.
"""
st = StandardScaler()

"""
# BITNO - StandardScaler fit() metoda se poziva samo na trening podacima. 
  Kada nauči mean i std na trening podacima, koristiće te vrednosti da transformiše i trening i test podatke
  To se radi da testni skup ne bi uticao na mean i std kojim se normalizuju trening podaci, što se može desiti ako bi se standardizacija radila na celom skupu podataka
  Testni skup treba da sadrži podatke koji su "nepoznati" modelu, i zbog toga želimo da izbegnemo da on utiče na bilo koji način na podatke na kojima treniramo
  jer to može lažno da poveća performanse modela na testnom skupu.
"""
st.fit(x_train)

"""
# BITNO - Nakon što se odradi fit, potrebno je samo transformisati trening podatke.
"""
x_train[x_train.columns] = st.transform(x_train[x_train.columns])

"""
# 8. Testni skup se deli na kolone pomoću kojih vršimo predikcije (X) i labele (Y).
"""
x_test = test.drop("zvanje", axis=1)
y_test = test['zvanje']

"""
# BITNO - na testnom skupu se koristi StandardScaler čija je funkcija fit prethodno pozvana na trening skupu.
  Sada se samo koriste mean i std koji su naučeni da bi se transformisao testni skup.
"""
x_test[x_test.columns] = st.transform(x_test[x_test.columns])


"""
# 9. Treniranje modela mašinskog učenja i vršenje predikcija.
# U scikitlearn biblioteci postoje gotovi modeli koji se mogu koristiti za rešavanje zadataka.
# Za klasifikaciju se može koristiti neuronska mreža MLPClassifier. Navedeni su neki od hiperparametara. Ostale možete pronaći u dokumentaciji modela.
  hidden_layer_size - broj neurona u svakom skrivenom sloju neuronske mreže. Podrazumevano ima jedan skriveni sloj sa 100 neurona
    Poželjno je dodati više sloja u neuronsku mrežu.
  max_iter - maksimalan broj epoha koliko model može da se trenira. Podrazumevano je 200.
    Poželjno je povećati broj epoha jer model ima sposobnost da se sam ranije zaustavi ukoliko ne može više da uči.
  learning_rate_init - brzina učenja. Podrazumevano je 0.001.
    Poželjno je malo uvećati ovu vrednost ukoliko model sporo uči.
  verbose - opcija da se ispisuju informacije o treniranju modela.
    Poželno je postaviti na True da bi stekli utisak o tome koliko brzo modle uči, i koliko mu je epoha potrebno da konvergira.
  random_state - random seed.
    Poželjno postaviti ukoliko želimo da pri svakom pokretanju dobijamo iste rezultate.
# Za sve modele važi da uče na trening skupu podataka. Treniranje se obavlja pomoću poziva fit funkcije sa trening podacima i labelama.
# Za sve modele važi da se predikcije vrše na testnom skupu. Poziva se predict metoda i prosleđuju se testni podaci.
# Poželjno je eskperimentisati sa drugačijim hiperparametrima modela, jer to može dovesti do boljih rezultata.
# Dozvoljeno je i korišćenje drugih modela.
  U kodu su zakomentarisani primeri nekih drugih modela.
"""
mlp = MLPClassifier(hidden_layer_sizes=[50,50,20],learning_rate_init=0.01, max_iter=1000,verbose=True, random_state=42).fit(x_train, y_train)
y_pred = mlp.predict(x_test)

"""
# 10. Iscrtavanje loss-a.
# Loss curve vam govori kako opada funkcija gubitka prilikom treniranja.
# Ona vam može dati uvid u to kako vaši hiperparametri utiču na treniranje.
"""
plt.plot(np.arange(len(mlp.loss_curve_)), mlp.loss_curve_)
plt.show()


# rf = RandomForestClassifier().fit(x_train, y_train)
# y_pred=  rf.predict(x_test)

# svc = SVC().fit(x_train, y_train)
# y_pred = svc.predict(x_test)

# model = ExtraTreesClassifier().fit(x_train, y_train)
# y_pred=  model.predict(x_test)

# model = AdaBoostClassifier().fit(x_train, y_train)
# y_pred=  model.predict(x_test)

# model = HistGradientBoostingClassifier().fit(x_train, y_train)
# y_pred=  model.predict(x_test)

# METRIKA
# y_pred -> predikcije vaseg modela
# y_test -> prave vrednosti iz csv
F1 = f1_score(y_pred, y_test, average='micro')
print(f'F1: {F1}')
