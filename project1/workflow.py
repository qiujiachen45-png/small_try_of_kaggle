from matplotlib.pyplot import figure
from numpy.ma.extras import row_stack
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.ensemble import  RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import  Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
from xgboost import XGBRegressor  # 专门用于回归任务的模型
import graphviz

#基础清理
df=pd.read_csv("../datasets/train.csv")
num=df.select_dtypes(include=np.number).columns.tolist()
word=df.select_dtypes(include=["object","category"]).columns.tolist()

num_transform=Pipeline([("Imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
word_transform=Pipeline([("imputer",SimpleImputer(strategy="constant",fill_value="missing"))])

df[num]=num_transform.fit_transform(df[num])
df[word]=word_transform.fit_transform(df[word])

preprocessor=ColumnTransformer(
    transformers=[("num",num_transform,num),("word",word_transform,word)],
    remainder="drop"
)

for i in df.select_dtypes("object"):
    df[i],_=df[i].factorize()

X=df.copy()
y=df.pop("SalePrice")



discrete_features=X.dtypes==int


def score_plot(X,y,discrete_features):
   mi_score=mutual_info_regression(X,y,discrete_features=discrete_features)
   mi_score=pd.DataFrame(mi_score,columns=["score"],index=X.columns)
   mi_score=mi_score.sort_values(by="score",ascending=False)
   return mi_score
the_chosen=score_plot(X,y, discrete_features)
the_chosen=the_chosen[:5]

#pca建模
# 1. 明确转换为 list
main_feature_names = the_chosen.head(10).index.tolist()

# 2. 筛选数值特征（PCA 必须是数值）
pca_cols = [f for f in main_feature_names if f in num and f !="SalePrice"]

# 3. 提取数据
main_feature= df[pca_cols]

main_features=(main_feature-main_feature.mean())/main_feature.std()

pca=PCA()



# --- 彻底修复这一段 ---
pca = PCA() # 如果不填参数，默认保留所有特征
X_pca = pca.fit_transform(main_features)

# 动态生成列名：确保 range 的长度和 X_pca 的列数 (shape[1]) 完全一致
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]

# 明确指定 data, columns 和 index
X_pca_df = pd.DataFrame(X_pca, columns=component_names, index=df.index)
# --------------------
def plot_drawing(pca):
    fig,(axis1,axis2)=plt.subplots(1,2,figsize=(4,12))
    n=pca.n_components_
    grid=np.arange(1,n+1)
    axis1.bar(grid,pca.explained_variance_ratio_)
    axis1.set(xlabel="component",title="方差占比图")
    axis2.plot(np.r_[0,grid],np.r_[0,np.cumsum(pca.explained_variance_ratio_)],"-o")
    axis2.set(xlabel="component",title="增长率")
    plt.show()

loading=pd.DataFrame(pca.components_.T,columns=component_names,index=pca_cols)
print(loading)

plot_drawing(pca)


kmeans=KMeans(n_clusters=5,n_init=12,random_state=0)
df["Cluster"]=kmeans.fit_predict(df[pca_cols])


# 将 PC1 和 Cluster 合并到展示数据中
final_preview = pd.concat([df[pca_cols], X_pca_df["PC1"], df["Cluster"]], axis=1)




X_enhance=df.copy()
X_enhance["PC1"]=X_pca_df["PC1"]
X_enhance["Cluster"]=df["Cluster"].astype("category")

X_enhance=pd.get_dummies(X_enhance)

model_xgb=xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    n_jobs=-1,
    random_state=42
)
scores=cross_val_score(
    model_xgb,X_enhance,y,
    cv=5,
    scoring="neg_mean_absolute_error"
)
mae_score=-scores

print("\n" + "="*30)
print("XGBoost 交叉验证实验结果")
print(f"5折平均 MAE: {mae_score.mean():.4f}")
print(f"标准差 (稳定性): {mae_score.std():.4f}")
print("="*30)


model_rf=RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    n_jobs=1,
    random_state=43,
)

rf_score=cross_val_score(
    model_rf,X_enhance,y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

mae_rf=-rf_score

print("\n" + "="*30)
print("随机森林 交叉验证结果")
print(f"5折平均 MAE: {mae_rf.mean():.4f}")
print(f"标准差: {mae_rf.std():.4f}")
print("="*30)
current_main_features=X_enhance.columns.tolist()
model_rf.fit(X_enhance,y)
single_tree = model_rf.estimators_[0]
tree_graph=tree.export_graphviz(single_tree,out_file=None,feature_names=current_main_features)
graphy=graphviz.Source(tree_graph)
graphy.view()
train_X, val_X, train_y, val_y = train_test_split(df, y, random_state=1)
disp1=PartialDependenceDisplay.from_estimator(single_tree,val_X,["OverallQual"])
plt.show()

fig,ax=plt.subplots(figsize=(4,8))
f_names=[("OverallQual","GrLivArea")]
disp4=PartialDependenceDisplay.from_estimator(single_tree,val_X,f_names,ax=ax)
plt.show()


import shap
row=2
data_for_predict=val_X.iloc[row]

explainer=shap.TreeExplainer(single_tree)
shap_value=explainer.shap_values(data_for_predict)

shap.initjs()
shap.force_plot(
    explainer.expected_value,   # 回归任务中这是一个标量（平均房价）
    shap_value,         # 这一行数据的特征贡献
    data_for_predict,           # 原始特征值
    matplotlib=True             # 保证在 PyCharm/脚本环境下能显示
)


# 1. 计算 SHAP 值
shap_values_all = explainer.shap_values(val_X)

# 2. 核心修正：如果结果是列表，取第一个元素（这是真正的贡献矩阵）
if isinstance(shap_values_all, list):
    shap_values_all = shap_values_all[0]

# 3. 绘图（此时 shap_values_all 变成了二维数组，可以正常显示了）
shap.summary_plot(shap_values_all, val_X)

# 1. 确保数据是二维矩阵（处理回归任务常见的列表包装问题）
s_values = shap_values_all[0] if isinstance(shap_values_all, list) else shap_values_all

# 2. 修正特征名称（使用你数据集中真实存在的列名）
shap.dependence_plot(
    "OverallQual",      # 你想观察的特征
    s_values,           # SHAP 值矩阵
    val_X,              # 对应的特征数据 (确保包含 OverallQual)
    interaction_index="GrLivArea"  # 颜色区分的交互特征
)
plt.title("SHAP Dependence Plot: OverallQual vs GrLivArea")
plt.tight_layout()
plt.show() # 强制弹出窗口



# 3. 显示图像
# 1. 读取测试集
test_df = pd.read_csv("../datasets/test.csv")

# 2. 保存 Id（最终提交只靠它）
test_ids = test_df["Id"]

# 3. 基础清理（和 train 一模一样）
test_num = test_df.select_dtypes(include=np.number).columns.tolist()
test_word = test_df.select_dtypes(include=["object", "category"]).columns.tolist()

# ⚠️ 使用【train 时 fit 过的】变换器
test_df[test_num] = num_transform.transform(test_df[test_num])
test_df[test_word] = word_transform.transform(test_df[test_word])

# 4. 类别特征 factorize（与 train 保持一致风格）
for col in test_df.select_dtypes("object"):
    test_df[col], _ = test_df[col].factorize()

# 5. PCA 特征（使用 train 拟合好的 pca）
test_main_feature = test_df[pca_cols]
test_main_features = (test_main_feature - main_feature.mean()) / main_feature.std()

X_test_pca = pca.transform(test_main_features)
X_test_pca_df = pd.DataFrame(
    X_test_pca,
    columns=component_names,
    index=test_df.index
)

# 6. KMeans 聚类（使用 train 拟合好的模型）
test_df["Cluster"] = kmeans.predict(test_df[pca_cols])

# 7. 构造增强特征（完全仿照 X_enhance）
X_test_enhance = test_df.copy()
X_test_enhance["PC1"] = X_test_pca_df["PC1"]
X_test_enhance["Cluster"] = test_df["Cluster"].astype("category")

X_test_enhance = pd.get_dummies(X_test_enhance)

# 8. 对齐列（防止 test / train one-hot 不一致）
X_test_enhance = X_test_enhance.reindex(
    columns=X_enhance.columns,
    fill_value=0
)

# 9. 预测
test_predictions = model_rf.predict(X_test_enhance)

# 10. 构造 submission（严格你给的格式）
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_predictions
})

# 11. 保存
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv 已生成，格式为：Id | SalePrice，可直接上交")