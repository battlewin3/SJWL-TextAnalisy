import pandas as pd
import numpy as np
from scipy.stats import pearsonr

data = pd.read_csv('binders_rates.csv')
#print(data.values)
datalist = data['English_words'].values
datadict = {}
for line in data.values:
    datadict[line[0]] = line
#print(datalist)

en = pd.read_csv('predict_en.csv')
#print(en.values)

en2 = []
data2 = []
for line in en.values:
    if line[0] in datalist:
        en2.append(list(line[1:]))
        data2.append(datadict[line[0]][1:])

en2 = np.array(en2)
data2 = np.array(data2)
print(en2.shape, data2.shape)
for i in range(8):
    print(pearsonr(en2[:,i], data2[:,i]))
