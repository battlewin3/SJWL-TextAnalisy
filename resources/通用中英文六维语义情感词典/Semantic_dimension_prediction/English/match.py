# 直到在中英文对齐词典中都有的词，现在已经通过extract_en从总库中提取出了中英文的词向量
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

dic_en_zh = {}
dic_zh_en = {}
classes_path = os.path.expanduser("cn_en_dict.txt")
with open(classes_path,'r',encoding = 'UTF-8') as f:
    s = f.readlines()
en = [c.strip().split('\t')[1] for c in s]
zh = [c.strip().split('\t')[0] for c in s]
for i in range(0,len(en)):
    # if(zh[i] == np.nan or en[i] == np.nan):
    #     continue
    dic_en_zh[en[i]] = zh[i]
    dic_zh_en[zh[i]] = en[i]
have_en = pd.read_csv('en_align.csv',header = None)
have_zh = pd.read_csv('zh_align.csv',header = None)
have_en = have_en.dropna(subset=[0])
en = list(have_en[0])
zh = list(have_zh[0])

en_have_zh = []
final_en = []
for i in en:
    en_have_zh.append(dic_en_zh[i])
final_zh = list(set(zh) & set(en_have_zh))
for i in final_zh:
    final_en.append(dic_zh_en[i])

have_en = have_en.loc[have_en[0].isin(final_en)]
have_zh = have_zh.loc[have_zh[0].isin(final_zh)]

# import pdb
# pdb.set_trace()

# align_en & align_zh是最终的已经对齐的中英文词
have_en[0] = have_en[0].astype('category')
have_en[0].cat.reorder_categories(final_en, inplace=True)
have_en.sort_values(0, inplace=True)
have_en.to_csv("align_en.csv",encoding='utf-8-sig',header = None,index=False)



have_zh[0] = have_zh[0].astype('category')
have_zh[0].cat.reorder_categories(final_zh, inplace=True)
have_zh.sort_values(0, inplace=True)
have_zh.to_csv("align_zh.csv",encoding='utf-8-sig',header = None,index=False)

