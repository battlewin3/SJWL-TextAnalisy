# 专门用于跑word2vec 因为有一些词里面没有
# 用来训练中文向量到标注的回归模型
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
warnings.filterwarnings('ignore')
s = []
wordvac = pd.read_csv('word2vec_train.csv', header = None)
wordvac = list(wordvac[0])
oup = pd.read_csv('feature.csv',encoding='utf-8-sig')
big = list(oup['word'])
a = list(set(big)-set(wordvac))
ss = StandardScaler()
oup = oup.loc[~oup.word.isin(a)] 
title = list(oup['word'])
col_oup = oup.columns.values.tolist()
col_oup.remove('word')
oup = oup[col_oup].copy()
files = ['word2vec_train.csv']
for i in files:
    pred = []
    print(i)
    inp = pd.read_csv(i,header = None, index_col = 0,encoding = 'utf-8-sig')
    X = inp.loc[title]
    X = ss.fit_transform(X)
    y = oup
    X = np.array(X)
    y = np.array(y)
    model = linear_model.Ridge()
    kflod=KFold(n_splits=10,shuffle=True)
    a = np.linspace(100,700,1000)
    param_grid = {'alpha':a}
    grid_search = GridSearchCV(model,param_grid,scoring = 'neg_mean_squared_error',n_jobs = -1,cv = kflod)
    grid_search.fit(X, y)
    print("Best: %f using %s" % (grid_search.best_score_,grid_search.best_params_))
    joblib.dump(grid_search.best_estimator_, 'train_decode.model')
