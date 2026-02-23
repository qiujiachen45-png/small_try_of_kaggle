import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

from workflow import kmeans, X_enhance

df=pd.read_csv("../datasets/train.csv")

num=df.select_dtypes(include=[np.number]).columns.tolist()
word=df.select_dtypes(include=["category","object"]).column.tolist()

num_transform1=Pipeline([("Imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
word_transform1=Pipeline([("Imputer",SimpleImputer(strategy="constant",fill_value="missing"))])

df[num]=num_transform1
df[word]=word_transform1

preprocessor=ColumnTransformer(transformers=[("num",num_transform1),("word",word_transform1)],remainder="drop")


for i in df.select_dtypes("object"):
    df[i],_=df[i].factorize()

X=df.copy()
y=X.pop("SalePrice")

discrete=X.dtype==int

def score_plot(X,y,discrete):
   mi_score=mutual_info_regression(X,y,discrete_features=discrete)
   mi_score=pd.DataFrame(mi_score,columns=["score"],index=index)
   mi_score=mi_score.sort_values(by="score",ascending=False)
   return mi_score


mi_score1=score_plot(X,y,discrete)
my_score=mi_score1[:5]

my_score=my_score.index.tolist()

pca2=[f for f in my_score if f in num and f != "SalaPrice"]

pca1=df[pca2]

main_features=(pca1-pca1.mean())/pca1.std()

pca=PCA()

pca_pca=pca.fit_transform(main_features)


components_name=[f"PC{i+1}" for i in range(pca_pca.shape[1])]

x_pca_df1=pd.DataFrame(pca_pca,columns=components_name,index=X.index)

def plot_drawing(pca):
    fig,(axis1,axis2)=plt.subplots(1,2,figsize=(4,12))
    n=pca.n_components_
    grid=np.arange(1,n+1)
    axis1.bar(grid,pca.explaned_variance_ratio_)
    axis1.set(xlabel="component",title="方差占比")
    axis2.plot(np.r_[0,gris],np.r_[0,pca.explaned_variance_ratio_],"-o")
    axis2.set(xlabel="component",title="增长率")
    plt.show()


loading=pd.DataFrame(pca.n_components_,columns=components_name,index=my_score)

kmeans=kmeans(n_cluster=5,n_init=5,random_state=0)
df["cluster"]=kmeans.fit_predict(df[my_score])

X_enhance1=df.copy()
X_enhance1["cluster"]=df["cluster"].astype("category")
X_enhance["PCA1"]=x_pca_df1["PC1"]


X_enhance1=pd.get_dummies(X_enhance)







