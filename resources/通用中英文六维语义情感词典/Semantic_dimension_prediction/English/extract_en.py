# 从很大的txt文本中抽取特定项
# 中文的抽取我在173那个服务器上做的，xinhua/extract
import os
import csv
import warnings
from nilearn import datasets, plotting
from nilearn.image import index_img, mean_img
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
from nilearn.plotting import plot_roi, plot_epi, show
from sklearn.cross_decomposition import PLSRegression
import seaborn as sns
import matplotlib.pyplot as plt
from  math  import  sqrt
import shutil
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

index = {}
classes_path = os.path.expanduser("cn_en_dict.txt")
with open(classes_path,'r',encoding = 'UTF-8') as f:
    s = f.readlines()
s = [c.strip().split('\t')[1] for c in s]

classes_path = os.path.expanduser("enwiki.model.txt")
with open(classes_path,'r',encoding = 'UTF-8') as f:
    txt = f.readlines()
v = [c.strip().split(' ')[0] for c in txt]
for i in range(0,len(s)):
    if(s[i] not in v):
        continue
    else:
        index[s[i]] = v.index(s[i])
txt = [c.strip() for c in txt]
print('txt')


for i in index.values():
   a = txt[i]
   a = a.split(' ')
   a = np.array(a)
   print('1')
   with open('en_align.csv','a+',encoding = 'utf-8-sig') as f:
       writer  = csv.writer(f)
       writer.writerow(a)