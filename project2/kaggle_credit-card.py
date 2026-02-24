
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from scipy.special.cython_special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.cluster import KMeans



db_train=pd.read_csv("datasets1/train.csv")
db_test=pd.read_csv("datasets1/test.csv")

#互相关探索的结果
#['Credit Default', 'Credit Score', 'Current Loan Amount', 'Term_Short Term', 'Years in current job_4 years']

#结果的概率分布
def analysis_of_feature_credit_default(db):
    plt.subplots(figsize=(4,4))
    sns.histplot(db["Credit Default"])
    plt.show()

#进行可视化，用可视化探究两个变量直接的关系和变量和结果的关系
def data_visuality_of_number(db):
    num_of_train=db.select_dtypes(include=[np.number]).columns.tolist()
    num_of_train.remove("Id")
    num_of_train.remove("Credit Default")
    #防止泄题
    plt.subplots((len(num_of_train)//3),3,figsize=(15,(((len(num_of_train)//3)+1)*5)))
    for i, feature in enumerate(num_of_train,1):
        plt.subplot((len(num_of_train)//3),3,i)
        sns.scatterplot(x=db["Credit Default"],y=db[feature],hue=db[feature],palette="Blues")

    plt.savefig("show_picture.png")
   

#进行数据清理（加强版）
def test_data_cleaning(db):
    num1=db.select_dtypes(include=[np.number]).columns.tolist()
    word1=db.select_dtypes(include=["object"]).columns.tolist()
    # 数值列移除，方便编码
    num1.remove("Credit Score")
    num1.remove("Current Loan Amount")

    # 字符列移除
    word1.remove("Term")
    word1.remove("Years in current job")

    #聚类分析（结果聚类）
    credit_score = db["Credit Score"].fillna(db["Credit Score"].median())
    kmeans_credit = KMeans(n_clusters=2, n_init=12, random_state=1)
    db["Cluster1"] = kmeans_credit.fit_predict(credit_score.values.reshape(-1, 1))
    db["Credit Score"] = credit_score

    #聚类分析（current load amount聚类）
    current_loan = db["Current Loan Amount"].fillna(db["Current Loan Amount"].median())
    kmeans_loan = KMeans(n_clusters=2, n_init=12, random_state=1)
    db["Cluster2"] = kmeans_loan.fit_predict(current_loan.values.reshape(-1, 1))
    db["Current Loan Amount"] = current_loan

    #聚类分析（Term）
    db["Term"] = db["Term"].map({"Short Term": 1, "Long Term": 0})
    kmeans_term = KMeans(n_clusters=2, n_init=12, random_state=1)
    db["Cluster3"] = kmeans_term.fit_predict(db["Term"].values.reshape(-1, 1))

    #mapping规则映射
    mapping = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10
    }
    #填充
    db["Years in current job"] = db["Years in current job"].map(mapping)
    db["Years in current job"]= db["Years in current job"].fillna( db["Years in current job"].mode()[0])
    
    for i in word1:
        db[i].fillna("NaN")
    for w in num1:
        db[w]=db[w].fillna(db[w].median())
    return db
#onehot编码文字
def coding_onehot(df1):
    print(df1.shape)
    word11 =['Home Ownership','Purpose']


    for i in word11:
    # 生成 dummy 列
       dummies = pd.get_dummies(df1[i], prefix=i, drop_first=True)

    # 删除原列，合并 dummy 列
       df1 = pd.concat([df1.drop(columns=[i]), dummies], axis=1)
    print(df1.shape)
    return df1
#正态分布
def box_change(db4):
    num1 = db4.select_dtypes(include=[np.number]).columns.tolist()
    db4[num1]=db4[num1].astype(float)

    for i in  num1:
        lamda=float(boxcox_normmax(db4[i].values + 1))
        db4[i]=boxcox1p(db4[i].values,lamda)
    return db4
#防止爆炸
def train_model(train_set1,test_set1):
    # 清理列名（去掉 [ ] < 等非法字符）
    train_set1.columns = [str(c).replace("[", "_").replace("]", "").replace("<", "_") for c in train_set1.columns]
    test_set1.columns = [str(c).replace("[", "_").replace("]", "").replace("<", "_") for c in test_set1.columns]

    # 分离特征和目标
    y_train = train_set1["Credit Default"]
    X_train = train_set1.drop(columns=["Credit Default"])
    X_test = test_set1.copy()

    # 测试集缺失的列补 0
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0

    # 保持列顺序一致
    #机器学习流程
    X_test = X_test[X_train.columns]
    X=X_train
    y=y_train
    model_lgb = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.05,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )
    model_lgb.fit(X,y)
    results=model_lgb.predict(X_test)
    results_binary = (results > 0.08).astype(int)
    submission1=pd.DataFrame({"Id":test_set1["Id"],"Credit Default":results_binary})
    submission1.to_csv("submission.csv",index=False)
    print(results.min(), results.max(), results.mean())
    print("OK")


db_train=data_cleaning(db_train)

db_train=coding_onehot(db_train)

db_test=test_data_cleaning(db_test)

db_test=coding_onehot(db_test)

train_model(db_train,db_test)
