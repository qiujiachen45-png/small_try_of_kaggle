import numpy as np
import pandas as pd
from pandas.conftest import axis_1
from sklearn.cluster import mean_shift

from PCA import X_pca

df=pd.read_csv("../datasets/train.csv")]

num_cols=df.select_dtypes(include=np.number).columns.tolist()
num_cols.remove("SalePrice")

X=df[num_cols].values

def standard(X):
    mean=X.means(axis=0)
    std=X.std(axis=0)
    return (X-mean)/std,mean,std]

X_std,X_mean,X_stddev=standard(X)

n=X.shape[0]

c=(X_std.T @ X_std)/n

eig_vals,eig_vecs=np.linalg.eig(c)


idx=np.argsort(eig_vals)[::-1]

X_pca=X_std @  W