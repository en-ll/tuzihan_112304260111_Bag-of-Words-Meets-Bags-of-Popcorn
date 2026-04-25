import re
import os

# 文本预处理函数
def preprocess_text(text):
    # 转小写
    text = text.lower()
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除标点和特殊符号
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 分词（使用正则表达式）
    tokens = re.findall(r'\b\w+\b', text)
    
    # 去停用词（使用内置的停用词列表）
    stop_words = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    ])
    tokens = [word for word in tokens if word not in stop_words]
    
    # 去除空字符串
    tokens = [word for word in tokens if word.strip()]
    
    return tokens

# 批量预处理函数
def preprocess_batch(texts):
    return [preprocess_text(text) for text in texts]
