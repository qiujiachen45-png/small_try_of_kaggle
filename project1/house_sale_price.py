
# 完整 Kaggle 房价预测脚本 (train + test + submission)


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb

# 1. 读取训练集

df = pd.read_csv("../datasets/train.csv")

# 保存特征类型
num = df.select_dtypes(include=np.number).columns.tolist()
word = df.select_dtypes(include=["object", "category"]).columns.tolist()


# 2. 分离 X / y

y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)


# 3. 数值和类别特征处理

# 排除 SalePrice
num_no_target = [c for c in num if c != "SalePrice"]

num_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
word_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing"))
])

# 数值列处理
X[num_no_target] = num_transform.fit_transform(X[num_no_target])
# 类别列处理
X[word] = word_transform.fit_transform(X[word])

# factorize 类别列
for col in X.select_dtypes(include=["object", "category"]):
    X[col], _ = X[col].factorize()


# 4. 特征选择（mutual_info_regression）

from sklearn.feature_selection import mutual_info_regression

discrete_features = X.dtypes == int
mi_score = mutual_info_regression(X, y, discrete_features=discrete_features)
mi_score = pd.DataFrame(mi_score, columns=["score"], index=X.columns)
mi_score = mi_score.sort_values(by="score", ascending=False)
the_chosen = mi_score.head(5).index.tolist()

# PCA 需要数值列
pca_cols = [f for f in the_chosen if f in num_no_target]
main_feature = X[pca_cols]
main_features_std = (main_feature - main_feature.mean()) / main_feature.std()

# 5. PCA

pca = PCA()
X_pca = pca.fit_transform(main_features_std)
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=component_names, index=X.index)

# 6. KMeans 聚类

kmeans = KMeans(n_clusters=5, n_init=12, random_state=0)
X["Cluster"] = kmeans.fit_predict(main_feature)

# 7. 构造增强特征 X_enhance

X_enhance = X.copy()
X_enhance["Cluster"] = X["Cluster"].astype("category")
for col in component_names:
    X_enhance[col] = X_pca_df[col]

# one-hot
X_enhance = pd.get_dummies(X_enhance)


# 8. 训练随机森林模型

model_xgb=xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    n_jobs=-1,
    random_state=42
)
model_xgb.fit(X_enhance, y)

# ===============================
# 9. 读取测试集
# ===============================
test_df = pd.read_csv("../datasets/test.csv")
test_ids = test_df["Id"]


# 10. 测试集基础清理
test_df[num_no_target] = num_transform.transform(test_df[num_no_target])
test_df[word] = word_transform.transform(test_df[word])

for col in test_df.select_dtypes(include=["object", "category"]):
    test_df[col], _ = test_df[col].factorize()


# 11. PCA

test_main_feature = test_df[pca_cols]
test_main_features_std = (test_main_feature - main_feature.mean()) / main_feature.std()
X_test_pca = pca.transform(test_main_features_std)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=component_names, index=test_df.index)


# 12. KMeam
test_df["Cluster"] = kmeans.predict(test_df[pca_cols])
X_test_enhance = test_df.copy()

# 13. 构造增强特征 X_test_enhance

# ';/X_test_enhance = test_df.copy()
X_test_enhance["Cluster"] = X_test_enhance["Cluster"].astype("category")
for col in component_names:
    X_test_enhance[col] = X_test_pca_df[col]

X_test_enhance = pd.get_dummies(X_test_enhance)
X_test_enhance = X_test_enhance.reindex(columns=X_enhance.columns, fill_value=0)


# 14. 预测 & submission

test_predictions = model_xgb.predict(X_test_enhance)
submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_predictions})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv 已生成，可直接提交")
