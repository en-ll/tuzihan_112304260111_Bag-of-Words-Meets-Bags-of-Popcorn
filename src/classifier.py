from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

# 训练随机森林分类器
def train_random_forest(X_train, y_train):
    print("训练随机森林分类器...")
    clf = RandomForestClassifier(
        n_estimators=200,  # 增加树的数量
        max_depth=20,      # 增加树的最大深度
        random_state=42,   # 随机种子
        n_jobs=-1          # 并行处理
    )
    clf.fit(X_train, y_train)
    print("随机森林分类器训练完成。")
    return clf

# 训练XGBoost分类器
def train_xgboost(X_train, y_train):
    print("训练XGBoost分类器...")
    clf = XGBClassifier(
        n_estimators=350,  # 最佳参数
        max_depth=13,      # 最佳参数
        learning_rate=0.05, # 最佳参数
        subsample=0.8,     # 最佳参数
        colsample_bytree=0.8, # 最佳参数
        random_state=42,   # 随机种子
        n_jobs=-1          # 并行处理
    )
    clf.fit(X_train, y_train)
    print("XGBoost分类器训练完成。")
    return clf

# 模型评估
def evaluate_model(clf, X_test, y_test):
    print("评估模型...")
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # 计算AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"模型AUC: {auc:.4f}")
    
    return auc

# 预测
def predict(clf, X):
    return clf.predict_proba(X)[:, 1]
