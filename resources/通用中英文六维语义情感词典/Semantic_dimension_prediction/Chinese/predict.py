# 生成基于word2vec的中文词
import os
import csv
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import linear_model
from  math  import  sqrt
import shutil
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from io import StringIO as StringIO
import re
warnings.filterwarnings('ignore')

model = joblib.load('train_decode_word2vec.model')

x = pd.read_csv('word2vec_predict.csv', header = None)
# x = x.loc[~x[0].isin(wordvac)]
title = list(x[0])
col_x = x.columns.values.tolist()
col_x.remove(0)
x = x[col_x].copy()
ss = StandardScaler()
x = ss.fit_transform(x)
y = model.predict(x)
time = 0
for j in y:
    with open('predict_zh.csv','a+',encoding = 'utf-8-sig') as f:
        writer  = csv.writer(f)
        writer.writerow(np.append(title[time],j))
        time = time + 1
test = pd.read_csv('predict_zh.csv',header = None)
test.columns = ['Word','Social','Vision','Motion','Space','Emotion','Time','Emotion_abs']
test.to_csv('predict_zh.csv',index=False,encoding = 'utf-8-sig')

