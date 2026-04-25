import numpy as np
from src.preprocess import preprocess_text

# 句子向量表示（平均方法）
def get_sentence_vector_avg(model, text):
    # 预处理文本
    tokens = preprocess_text(text)
    
    # 收集词向量
    word_vectors = []
    for word in tokens:
        if word in model.wv:
            word_vectors.append(model.wv[word])
    
    # 计算平均向量
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        # 如果没有找到任何词向量，返回零向量
        return np.zeros(model.vector_size)

# 批量获取句子向量
def get_sentence_vectors_avg(model, texts):
    return np.array([get_sentence_vector_avg(model, text) for text in texts])

# 句子向量表示（TF-IDF加权平均方法）
def get_sentence_vector_tfidf(model, text, tfidf_vectorizer):
    # 预处理文本
    tokens = preprocess_text(text)
    
    # 计算TF-IDF权重
    tfidf_vector = tfidf_vectorizer.transform([text])
    
    # 收集词向量和对应的权重
    word_vectors = []
    weights = []
    for word in tokens:
        if word in model.wv and word in tfidf_vectorizer.vocabulary_:
            word_idx = tfidf_vectorizer.vocabulary_[word]
            weight = tfidf_vector[0, word_idx]
            if weight > 0:
                word_vectors.append(model.wv[word])
                weights.append(weight)
    
    # 计算加权平均向量
    if word_vectors:
        return np.average(word_vectors, axis=0, weights=weights)
    else:
        # 如果没有找到任何词向量，返回零向量
        return np.zeros(model.vector_size)

# 批量获取TF-IDF加权句子向量
def get_sentence_vectors_tfidf(model, texts, tfidf_vectorizer):
    return np.array([get_sentence_vector_tfidf(model, text, tfidf_vectorizer) for text in texts])
