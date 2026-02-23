
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression


plt.style.use("seaborn-v0_8-whitegrid")

df=pd.read_csv("../datasets/train.csv")

num=df.select_dtypes(include=[np.number]).columns.tolist()
word=df.select_dtypes(include=["object","category"]).columns.tolist()

num_impute=SimpleImputer(strategy="median")
df[num]=num_impute.fit_transform(df[num])

word_impute=SimpleImputer(strategy="constant",fill_value="missing")
df[word]=word_impute.fit_transform(df[word])

X=df.copy()
y=X.pop("SalePrice")

for i in X.select_dtypes(include="object"):
    X[i],_=X[i].factorize()




discrete_feature=X.dtypes==int

def plot_o(X,y,discrete_feature):
    m_score=mutual_info_regression(X,y,discrete_features=discrete_feature)
    m_score = pd.DataFrame(m_score, columns=["score"], index=X.columns)
    m_score = m_score.sort_values(by="score", ascending=False)
    return m_score

f_score=plot_o(X,y, discrete_feature)
print(f_score[:20])


sns.relplot(x="SalePrice",y="OverallQual",data=df)
plt.show()
sns.relplot(x="SalePrice",y="Neighborhood",data=df)
plt.show()

