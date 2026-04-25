import os
import pandas as pd
from gensim.models import Word2Vec
from src.preprocess import preprocess_batch

# 读取数据
def load_data():
    # 读取有标签训练数据
    labeled_train = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)
    
    # 读取无标签训练数据
    unlabeled_train = pd.read_csv('unlabeledTrainData.tsv/unlabeledTrainData.tsv', sep='\t', quoting=3)
    
    # 读取测试数据
    test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)
    
    return labeled_train, unlabeled_train, test_data

# 训练Word2Vec模型
def train_word2vec():
    # 加载数据
    labeled_train, unlabeled_train, _ = load_data()
    
    # 合并所有文本数据用于训练
    all_texts = pd.concat([labeled_train['review'], unlabeled_train['review']], ignore_index=True)
    
    # 预处理文本
    print("预处理文本...")
    tokenized_texts = preprocess_batch(all_texts)
    
    # 训练Word2Vec模型
    print("训练Word2Vec模型...")
    model = Word2Vec(
        tokenized_texts,
        vector_size=300,  # 词向量维度
        window=5,         # 上下文窗口大小
        min_count=5,      # 最小词频
        workers=4,        # 并行处理线程数
        epochs=10         # 训练轮数
    )
    
    # 保存模型
    model.save('word2vec.model')
    print("Word2Vec模型训练完成并保存。")
    
    return model

# 加载Word2Vec模型
def load_word2vec():
    if os.path.exists('word2vec.model'):
        print("加载已训练的Word2Vec模型...")
        model = Word2Vec.load('word2vec.model')
        print("Word2Vec模型加载完成。")
    else:
        print("未找到已训练的Word2Vec模型，开始训练...")
        model = train_word2vec()
    
    return model

# 获取词向量
def get_word_vector(model, word):
    if word in model.wv:
        return model.wv[word]
    else:
        return None
