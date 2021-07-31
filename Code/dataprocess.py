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
def equidistance_select(input, output, num_):
    result_sentence = []
    remain_sentence = []
    count = -1
    for line in input:
        count = count + 1
        if len(result_sentence) >= 100:
                remain_sentence.append(line)
        else:
            if count % num_ == 0:        #调节间距，根据数据总数，等距取数据
                result_sentence.append(line)
            else:
                remain_sentence.append(line)

#将去除预训练100条句子的剩余句子提取出来
    for line in remain_sentence:
        output.write(line)
    return result_sentence


if __name__ == '__main__':

    ##获取训练数据
    inputs1 = open('result/data_origin.txt', 'r', encoding='utf-8')
    outputs1 = open('result/train_data_pre_100.txt', 'w', encoding='utf-8')

    outputs2 = open('result/train_data_pre_remain.txt', 'w', encoding='utf-8')
    inputs_fin1 = equidistance_select(inputs1, outputs2, num_= 7)


    for line in inputs_fin1:
        line = trim(line)
        line_seg = segment_sentence(line)
        outputs1.write(line_seg + '\n')
    outputs1.close()
    inputs1.close()
    outputs2.close()

    ##获取测试数据
    inputs2 = open('Data/data_origin.txt.txt', 'r', encoding='utf-8')
    outputs3 = open('result/test_data_pre_100.txt', 'w', encoding='utf-8')

    outputs4 = open('result/train_data_pre_remain_remain.txt', 'w', encoding='utf-8')
    inputs_fin2 = equidistance_select(inputs2, outputs4, num_= 6)

    for line in inputs_fin2:
        line = trim(line)
        line_seg = segment_sentence(line)
        outputs3.write(line_seg + '\n')
    outputs3.close()
    inputs2.close()