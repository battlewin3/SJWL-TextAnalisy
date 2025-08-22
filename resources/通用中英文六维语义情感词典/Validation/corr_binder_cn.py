import pandas as pd
import numpy as np
from scipy.stats import pearsonr

data = pd.read_csv('binders_rates_cn.csv')
#print(data.values)
datalist = data['Chinese_translations'].values
datadict = {}
for line in data.values:
    datadict[line[0]] = line
print(datalist[0])

en = pd.read_csv('predict_zh.csv')
#print(en.values)

en2 = []
data2 = []
for line in en.values:
    if line[0] in datalist:
        print(line[0])
        en2.append(list(line[1:])])
        data2.append(datadict[line[0]][1:])

en2 = np.array(en2)
data2 = np.array(data2)
print(en2.shape, data2.shape)
for i in range(8):
    print(pearsonr(en2[:,i], data2[:,i]))
