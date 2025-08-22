# 将英文转为中文之后，生成相应的词向量，并预测feature
import os
import csv
import warnings
from nilearn import datasets, plotting
from nilearn.image import index_img, mean_img
import joblib
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
from  math  import  sqrt
import shutil
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from io import StringIO as StringIO
import re
warnings.filterwarnings('ignore')


model = joblib.load('align/train_align.model')
x = pd.read_csv('word2vec_predict.csv',header = None)
title = list(x[0])
col_x = x.columns.values.tolist()
col_x.remove(0)
x = x[col_x].copy()
ss = StandardScaler()
x = ss.fit_transform(x)
x = model.predict(x)
x = ss.fit_transform(x)
model = joblib.load('train_decode.model')
y = model.predict(x)
time = 0
for j in y:
    with open('predict_en.csv','a+',encoding = 'utf-8-sig') as f:
        writer  = csv.writer(f)
        writer.writerow(np.append(title[time],j))
        time = time + 1

test = pd.read_csv('predict_en.csv',header = None)
test.columns = ['Word','Social','Vision','Motion','Space','Emotion','Time','Emotion_abs']
test.to_csv('predict_en.csv',index=False,encoding = 'utf-8-sig')

