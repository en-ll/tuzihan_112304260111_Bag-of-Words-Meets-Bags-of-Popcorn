import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# 加载保存的句子向量
def load_vectors():
    X = np.load('X.npy')
    y = np.load('y.npy')
    return X, y

# 网格搜索优化XGBoost参数
def grid_search_xgboost():
    print("加载数据...")
    X, y = load_vectors()
    
    print("设置网格搜索参数...")
    # 定义参数网格（减少参数组合数量）
    param_grid = {
        'n_estimators': [300, 350],
        'max_depth': [12, 13],
        'learning_rate': [0.05],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # 创建XGBoost分类器
    xgb = XGBClassifier(random_state=42, n_jobs=4)  # 限制线程数
    
    # 创建网格搜索对象
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=2,  # 减少交叉验证折数
        n_jobs=1,  # 串行处理，减少内存使用
        verbose=2
    )
    
    print("开始网格搜索...")
    # 执行网格搜索
    grid_search.fit(X, y)
    
    # 打印最佳参数
    print("最佳参数:")
    print(grid_search.best_params_)
    
    # 打印最佳AUC分数
    print("最佳AUC分数:")
    print(grid_search.best_score_)
    
    return grid_search.best_estimator_

if __name__ == "__main__":
    best_model = grid_search_xgboost()
