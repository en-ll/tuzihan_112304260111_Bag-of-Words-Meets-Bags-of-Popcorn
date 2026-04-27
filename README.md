# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：屠子涵
- **学号**：112304260111
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview/part-1-for-beginners-bag-of-words
- **提交日期**：2026-04-27

- **GitHub 仓库地址**：https://github.com/en-ll/tuzihan_112304260111_Bag-of-Words-Meets-Bags-of-Popcorn
- **GitHub README 地址**：https://github.com/en-ll/tuzihan_112304260111_Bag-of-Words-Meets-Bags-of-Popcorn/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.94859
- **Private Score**（如有）：0.94859
- **排名**（如能看到可填写）：

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./112304260111_屠子涵_kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**
- 转小写：将所有文本转换为小写
- 去除HTML标签：使用正则表达式去除文本中的HTML标签
- 保留否定词：将否定词与后面的词组合，形成短语模式（例如："not good" -> "not_good"）
- 去除标点和特殊符号：使用正则表达式去除非字母，空格和下划线的字符
- 分词：使用正则表达式进行分词，支持短语模式（如下划线连接的词）
- 去停用词：使用内置的英文停用词列表去除停用词（保留否定词如not）
- 去除空字符串：过滤掉分词后产生的空字符串

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**
- 自己训练Word2Vec模型：使用有标签和无标签的训练数据进行训练
- 词向量维度：300维
- 句子向量表示：使用平均方法，将句子中所有词的词向量取平均值作为句子向量

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**
使用了逻辑回归分类模型，参数设置如下：
- C=1.0（正则化强度）
- random_state=42（随机种子）

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集
2. 对文本进行预处理
3. 训练或加载 Word2Vec 模型
4. 将每条文本表示为句向量
5. 用训练集训练分类器
6. 在测试集上预测结果
7. 生成 submission 文件并提交 Kaggle

**我的实验流程：**
1. 读取有标签训练数据、无标签训练数据和测试数据
2. 对所有文本数据进行预处理，包括转小写、去除HTML标签、保留否定词与后面词的组合（如not_good）、去除标点、分词和去停用词
3. 使用预处理后的文本训练Word2Vec模型
4. 将每条文本转换为句向量（使用平均方法）
5. 划分训练集和验证集
6. 使用训练集训练逻辑回归分类器
7. 在验证集上评估模型，计算AUC
8. 在测试集上预测结果
9. 生成Kaggle提交文件

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

示例：
- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**
```text
project/
├─ labeledTrainData.tsv/：存放有标签训练数据
├─ unlabeledTrainData.tsv/：存放无标签训练数据
├─ testData.tsv/：存放测试数据
├─ src/：存放源代码
│  ├─ preprocess.py：文本预处理功能
│  ├─ word2vec.py：Word2Vec模型训练和加载
│  ├─ sentence_vector.py：句子向量表示
│  └─ classifier.py：分类模型训练和评估
├─ images/：存放README中使用的图片
├─ main.py：主脚本
├─ requirements.txt：Python依赖
├─ .env：环境变量
├─ .env.example：环境变量模板
├─ .gitignore：Git忽略文件
└─ README.md：项目说明
```
