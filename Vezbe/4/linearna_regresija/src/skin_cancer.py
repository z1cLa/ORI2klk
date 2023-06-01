from __future__ import print_function
import csv
import codecs
import linreg_simple as lrs
import matplotlib.pyplot as plt

# Funkcija za citanje .csv fajla (comma separated file)
def read_csv_file(filepath, geotype='Long'):
    x = []
    y = []

    index = 4 if geotype == "Long" else 1

    with open(filepath, 'r') as f:
        read = csv.reader(f, delimiter=',')
        for i, row in enumerate(read):
            if i == 0:  # skip header
                continue
            x.append(float(row[index]))
            y.append(int(row[2]))

    return x, y


# Funkcija da bismo mogli lakse pozvati i za geografsku duzinu i za geografsku sirinu
def linreg_cancer(geotype):
    x, y = read_csv_file('data/skincancer.csv', geotype)

    slope, intercept = lrs.fit(x, y)

    line_y = lrs.make_predictions(x, slope, intercept)

    print("grafik")
    plt.plot(x, y, '.')
    plt.plot(x, line_y, 'r')
    plt.title('Slope: {0}, intercept: {1}'.format(slope, intercept))
    plt.show()


if __name__ == '__main__':
    linreg_cancer('Lat')
    linreg_cancer('Long')
