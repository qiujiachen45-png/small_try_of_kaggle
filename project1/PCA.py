


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression




from sklearn.feature_selection import mutual_info_regression

plt.style.use("seaborn-v0_8-whitegrid")

plt.rc("figure",autolayout=True)

plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight=14,
    titlepad=10
)


def plot_variance(pca, width=8, dpi=100):
    # 创建画布 (注意 subplots 有 s)
    fig, axs = plt.subplots(1, 2, figsize=(width, 4), dpi=dpi)

    # 获取组件数量 (注意拼写 n_components_)
    n = pca.n_components_
    grid = np.arange(1, n + 1)

    # 绘制单个解释方差图
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))

    # 绘制累积方差图 (注意使用 .index)
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0))


    plt.show()
    return axs


def make_mi_score(X,y,discrete_features):
    mi_score=mutual_info_regression(X,y,discrete_features=discrete_features)
    mi_score=pd.Series(mi_score,name="MI Scores", index=X.columns)
    mi_score=mi_score.sort_values(ascending=False)
    return mi_score
df=pd.read_csv("../datasets/train.csv")

num=df.select_dtypes(include=[np.number]).columns.tolist()
word=df.select_dtypes(include=["object","category"]).columns.tolist()
num_impute=SimpleImputer(strategy="median")
df[num]=num_impute.fit_transform(df[num])

word_impute=SimpleImputer(strategy="constant",fill_value="missing")
df[word]=word_impute.fit_transform(df[word])

features=["LotFrontage","MSSubClass","GrLivArea"]

X=df.copy()
y=X.pop("SalePrice")
X=X.loc[:,features]


X_scaled=(df[num]-df[num].mean(axis=0))/df[num].std(axis=0)


from sklearn.decomposition import  PCA

pca=PCA()
X_pca=pca.fit_transform(X_scaled)

component_name=[f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca=pd.DataFrame(X_pca,columns=component_name)

X_pca.head()

idx=X_pca["PC3"].sort_values(ascending=False).index
cols=["OverallQual","Neighborhood","GrLivArea"]
df.loc[idx,cols]


df["sports_or_wagon"]=X.MSSubClass/X.LotFrontage
sns.regplot(x="LotFrontage",y="MSSubClass",data=df,order=2)



plot_variance(pca);
