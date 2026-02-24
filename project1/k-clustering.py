#对"OverallQual","Neighborhood","GrLivArea"进行聚类分析画图
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure",autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

df=pd.read_csv("../datasets/train.csv")
num=df.select_dtypes(include=[np.number]).columns.tolist()
word=df.select_dtypes(include=["object","category"]).columns.tolist()

num_impute=SimpleImputer(strategy="median")
df[num]=num_impute.fit_transform(df[num])

word_impute=SimpleImputer(strategy="constant",fill_value="missing")
df[word]=word_impute.fit_transform(df[word])



X=df.loc[:,["OverallQual","Neighborhood","GrLivArea"]]

for i in X.select_dtypes(include="object"):
    X[i],_=X[i].factorize()

KMeans=KMeans(n_clusters=6)
X["cluster"]=KMeans.fit_predict(X)
X["cluster"]=X["cluster"].astype("category")

sns.relplot(x="OverallQual",y="Neighborhood",hue="GrLivArea",data=X,height=6)

plt.show()

