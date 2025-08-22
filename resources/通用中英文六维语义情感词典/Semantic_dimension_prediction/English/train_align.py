# 用来训练从英文到中文的模型

# 专门用于跑bert 因为有一些词里面没有
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
X = pd.read_csv('align_en.csv', header = None, index_col = 0)
oup = pd.read_csv('align_zh.csv',encoding='utf-8-sig',header = None, index_col = 0)
# big = list(oup['word'])
# a = list(set(big)-set(wordvac))
# classes_path = os.path.expanduser("/home/yhzhang/RSA/word_dict.txt")
# with open(classes_path,'r',encoding = 'UTF-8') as f:
#     s = f.readlines()
# s = [c.strip() for c in s]
ss = StandardScaler()
X = ss.fit_transform(X)
y = oup
# loo = LeaveOneOut()
# kf = KFold(n_splits=10)
X = np.array(X)
y = np.array(y)
model = linear_model.Ridge()
kflod=KFold(n_splits=10,shuffle=True)
a = np.linspace(4000,6000,1000)
param_grid = {'alpha':a}
grid_search = GridSearchCV(model,param_grid,scoring = 'neg_mean_squared_error',n_jobs = -1,cv = kflod)
grid_search.fit(X, y)
print("Best: %f using %s" % (grid_search.best_score_,grid_search.best_params_))
joblib.dump(grid_search.best_estimator_, 'train_align.model')
