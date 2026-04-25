import pandas as pd
from src.word2vec import load_data, load_word2vec
from src.sentence_vector import get_sentence_vectors_avg
from src.classifier import train_random_forest, evaluate_model, predict
from sklearn.model_selection import train_test_split

# 主函数
def main():
    print("=== Bag of Words Meets Bags of Popcorn ===")
    
    # 1. 加载数据
    print("加载数据...")
    labeled_train, unlabeled_train, test_data = load_data()
    
    # 2. 加载或训练Word2Vec模型
    model = load_word2vec()
    
    # 3. 生成句子向量（平均方法）
    print("生成句子向量...")
    X = get_sentence_vectors_avg(model, labeled_train['review'])
    y = labeled_train['sentiment'].values
    
    # 4. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. 训练随机森林分类器
    clf = train_random_forest(X_train, y_train)
    
    # 6. 评估模型
    auc = evaluate_model(clf, X_val, y_val)
    
    # 7. 生成测试集预测
    print("生成测试集预测...")
    X_test = get_sentence_vectors_avg(model, test_data['review'])
    test_predictions = predict(clf, X_test)
    
    # 8. 生成Kaggle提交文件
    print("生成Kaggle提交文件...")
    # 确保id列的值没有多余的引号
    test_data['id'] = test_data['id'].str.strip('"')
    submission = pd.DataFrame({
        'id': test_data['id'],
        'sentiment': test_predictions
    })
    submission.to_csv('submission.csv', index=False, quoting=0)
    print("提交文件已生成: submission.csv")
    
    print("\n=== 任务完成 ===")
    print(f"模型AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
