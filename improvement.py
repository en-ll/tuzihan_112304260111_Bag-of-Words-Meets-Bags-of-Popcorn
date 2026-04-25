import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.word2vec import load_data, load_word2vec
from src.sentence_vector import get_sentence_vectors_tfidf
from src.classifier import train_random_forest, evaluate_model, predict
from sklearn.model_selection import train_test_split

# 使用TF-IDF加权平均方法改进
def main_tfidf():
    print("=== Bag of Words Meets Bags of Popcorn (TF-IDF加权) ===")
    
    # 1. 加载数据
    print("加载数据...")
    labeled_train, unlabeled_train, test_data = load_data()
    
    # 2. 加载或训练Word2Vec模型
    model = load_word2vec()
    
    # 3. 训练TF-IDF向量器
    print("训练TF-IDF向量器...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,  # 最大特征数
        stop_words='english'  # 去停用词
    )
    # 使用所有文本训练TF-IDF
    all_texts = pd.concat([labeled_train['review'], unlabeled_train['review']], ignore_index=True)
    tfidf_vectorizer.fit(all_texts)
    
    # 4. 生成句子向量（TF-IDF加权平均方法）
    print("生成句子向量...")
    X = get_sentence_vectors_tfidf(model, labeled_train['review'], tfidf_vectorizer)
    y = labeled_train['sentiment'].values
    
    # 5. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. 训练随机森林分类器
    clf = train_random_forest(X_train, y_train)
    
    # 7. 评估模型
    auc = evaluate_model(clf, X_val, y_val)
    
    # 8. 生成测试集预测
    print("生成测试集预测...")
    X_test = get_sentence_vectors_tfidf(model, test_data['review'], tfidf_vectorizer)
    test_predictions = predict(clf, X_test)
    
    # 9. 生成Kaggle提交文件
    print("生成Kaggle提交文件...")
    submission = pd.DataFrame({
        'id': test_data['id'],
        'sentiment': test_predictions
    })
    submission.to_csv('submission_tfidf.csv', index=False)
    print("提交文件已生成: submission_tfidf.csv")
    
    print("\n=== 任务完成 ===")
    print(f"模型AUC: {auc:.4f}")

if __name__ == "__main__":
    main_tfidf()
