
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from fontTools.misc.arrayTools import scaleRect
from hypothesis.internal.coverage import pretty_file_name_cache
from pure_eval.utils import safe_name_samples
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
from sklearn.pipeline import  Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA


#基础清理
df=pd.read_csv("../datasets/train.csv")
num=df.select_dtypes(include=np.number).columns.tolist()
word=df.select_dtypes(include=["object","category"]).columns.tolist()
#数据清理管道
#num清理，word的编码
num_transform=Pipeline([("Imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
word_transform=Pipeline([("imputer",SimpleImputer(strategy="constant",fill_value="missing")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])

df[num]=num_transform.fit_transform(df[num])
df[word]=word_transform.fit_transform(df[word])

X=df.copy()
y=df.pop("SalePrice")
#防止数据渗漏
for i in X.select_dtypes("object"):
    X[i],_=X[i].factorize()

discrete_features=X.dtypes==int

#互信息筛选前十变量
def score_plot(X,y,discrete_features):
   mi_score=mutual_info_regression(X,y,discrete_features=discrete_features)
   mi_score=pd.DataFrame(mi_score,columns=["scores"],index=X.columns)
   mi_score=mi_score.sort_values(by="score",ascending=False)
   return mi_score
the_chosen=score_plot(X,y, discrete_features)
the_top_feature=the_chosen.head(10).index
the_chosen1=df[the_top_feature].select_dtypes(include=[np.number]).columns.tolist()


the_chosen1=df[the_chosen1]
scaler=StandardScaler()
X_pca_scaled=scaler.fit_transform(the_chosen1)
降维为三维
pca=PCA(n_components=3)
X_pca=pca.fit_transform(X_pca_scaled)
X_pca_df=pd.DataFrame(X_pca,columns=["PC1","PC2"],index=df.index)


kmeans=KMeans(n_clusters=5,random_state=0,n_init=15)
df["Cluster"]=kmeans.fit_predict(X_pca)




