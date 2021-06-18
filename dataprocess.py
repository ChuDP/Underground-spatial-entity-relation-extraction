"""
该代码的目的是对命名实体识别的结果文件进行处理，将挑选出一个句子中有两个GEO的句子
"""

import pickle
from collections import Counter
import pandas as pd
import logging
import multiprocessing
import os.path
import sys
import jieba
import jieba.posseg as pseg
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

################## 加载自定义词典######################
jieba.load_userdict("关系提取实体库_去重_可扩展.txt")
jieba.load_userdict("地质专业词汇词典.txt")
jieba.load_userdict("常用词词典.txt")
jieba.load_userdict("去重标注库.txt")
jieba.load_userdict("关系提取实体库.txt")
jieba.load_userdict("关系提取关系词库.txt")

###############读取识别结果数据################
def load_data():
    all_w, all_label = [], []
    from glob import glob
    for file in glob('data/data.csv'):
        df = pd.read_csv(file, sep=',')
        all_w += df['word'].tolist()
        all_label += df['label'].tolist()
    print(all_w)
    print(all_label)
    num_samples = len(df)
    print(num_samples)
    sep_index = [-1] + df[df['word'] == 'sep'].index.tolist() + [num_samples]  # 20 40 50
    print(sep_index)

    words = []
    labels = []
    for i in range(len(sep_index) - 1):
        start = sep_index[i] + 1
        end = sep_index[i + 1]
        count = 0
        for feature in df.columns:
            if feature == 'word':
                count += 1
            if count == 1:
                words.append(list(df[feature])[start:end])
                count = 0
            if feature == 'label':
                labels.append(list(df[feature])[start:end])

######################将二维列表中的分词结果进行合并形成句子######################
    s = ""
    sentences = []
    for sentence in words:
        for word in sentence:
            s = s + str(word)
        sentences.append(s)
        s = ""

    num1 = 0
    num2 = 0
    result_sen = []
    for label in labels:
        num1 += 1
        for i in label:
            if i == 'B-GEO':
                num2 += 1
        if num2 >= 2:
            result_sen.append(sentences[num1-1] + '\n')
            num2 = 0
    print(result_sen)
    with open("data/训练数据1.txt", "w", encoding="UTF-8") as resultFile:
         resultFile.writelines(result_sen)


######################分词函数############################
# 分词函数
def segment_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    outstr = ''
    for word in sentence_seged:
        outstr += word
        outstr += " "
    return outstr

#########################去除空格函数#########################
def trim(sentence):
    sentence = sentence.replace(' ', '')
    return sentence


#######################等距选择样本，共选择100个句子###########################
def equidistance_select(input):
    result_sentence = []
    remain_sentence = []
    count = -1
    for line in input:
        count = count + 1
        if count % 6 == 0:
            result_sentence.append(line)
            if len(result_sentence) >= 100:
                break
        else:
            remain_sentence.append(line)

#将去除预训练100条句子的剩余句子提取出来
    # for line in remain_sentence:
    #     output.write(line)
    return result_sentence


if __name__ == '__main__':
    # load_data()
    inputs = open('result/train_data_pre.txt', 'r', encoding='utf-8')
    # outputs2 = open('result/train_data_pre_remain.txt', 'w', encoding='utf-8')
    # inputs_fin =equidistance_select(inputs)
    outputs1 = open('result/train_data_pre_100_all.txt', 'w', encoding='utf-8')

    for line in inputs:
        line = trim(line)
        line_seg = segment_sentence(line)
        outputs1.write(line_seg + '\n')
    outputs1.close()
    inputs.close()
