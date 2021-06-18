"""
本节代码的作用是提取Bootstrapping各项特征系数
本实验选择的特征包括（1)词性特征 （2）关系词所处的位置 （3）句法依存关系 （4）左边有连词或者介词时的位置 （5）到e1的距离 （6）到e2的距离 （7）关系词的长度 （8）e1到e2的距离
"""
import random
from collections import Counter
import pickle
import pandas as pd
import jieba
import jieba.posseg as pseg
import itertools
import numpy as np
from dataprocess import trim, segment_sentence

################## 加载自定义词典######################
jieba.load_userdict("地质专业词汇词典.txt")
jieba.load_userdict("常用词词典.txt")
jieba.load_userdict("去重标注库.txt")
jieba.load_userdict("关系提取实体库_quchong.txt")
jieba.load_userdict("关系提取关系词库_quchong.txt")
jieba.load_userdict("Relation_CX_flag.txt")
jieba.load_userdict("关系提取实体库.txt")
jieba.load_userdict("关系提取关系词库.txt")
data={} #用来存放数据的字典
######################读取等距读取的100个句子，以整句的形式进行存储########################
def data_load1(inputs):
    a1 = []
    for line in inputs:
        line = line.replace(' ', '').strip('\n')
        a1.append(line)
    print(a1)
    return a1

##############读取等距读取的100个句子，以二维列表的形式存储，方便后面进行标注################
def data_load2(inputs):
    ss = []           ########存储所有句子
    for lines in inputs:
        line = lines.strip().split(' ')
        s = []        ######存储以空格分开的单个句子
        for i in line:
             s.append(i)
        ss.append(s)
    return ss


##################实体去重####################
def quchong(infile,outfile):
    inputs = open(infile, 'r', encoding='utf-8')
    outputs = open(outfile, 'w', encoding='utf-8')

    List_row = inputs.readlines()
    list_source = []
    for list_line in List_row:

        list_line = list_line.strip().split(' ')
        list_source.append(list_line)
    # print(len(list_source))
    # print(list_source)

    list1 = list(set([tuple(t) for t in list_source]))
    dic = sorted([list(v) for v in list1], key=list_source.index)
    ######################一维列表写入文本####################
    for i in range(len(dic)):
        for j in range(len(dic[i])):
            outputs.write(str(dic[i][j]))
        outputs.write('\n')
    outputs.close()
    return dic

######标注实体，将一个句子中所有的地质实体标注出来，按照entity1,entity2...entityn的形式进行标注######
########(后续存在一个问题就是中文分词会出现错误的情况，这个肯定会影响精度，本实验不解决这个问题)
def biaozhu_entity(cankaoku,infile,outputfile):
    shiticankao = open(cankaoku, 'r', encoding='utf-8')
    a1 = []
    for line in shiticankao:
        line = line.replace(' ', '').strip('\n')
        a1.append(line)
    print(a1)                             #########a1表示读取后的实体参考库

    # relationcankao = open(relationKu, 'r', encoding='utf-8')
    # R1 = []
    # for line in relationcankao:
    #     line = line.replace(' ', '').strip('\n')
    #     R1.append(line)
    # print(R1)  #########R1表示读取后的关键词参考库
    yuanwen = data_load2(infile)

    #####################根据实体对数复制句子######################
    sentences_copy = []                      #存放复制后的所有句子
    flag_all = []                            #存放所有句子的标注结果
    CX_all_tag = []                          #存放所有句子词性标注结果
    entity_pairs = []                        #存放所有句子的实体对
    position_index_pairs = []                #存放所有的实体对位置索引
    temp_sentences = []                      #存放列表元素合并后的所有句子
    temp_sentence = []                       # 存放单个列表元素合并后的句子

    for i in yuanwen:
        flag_one = []                     #存放单个句子关系词和实体标注结果
        entity_count = 1
        flag_entity = []                  #存放后续的实体标注
        entity_orign = []                 #用于存放被标注的实体本体
        entity_pair = []                  #存放单个句子的实体对
        sentence_copy = []                #存放单个句子的实体对
        position_index = []               #存放单个实体对的位置索引
        position_index_pair = []          #存放一个句子中实体对的相应位置
        Cx_one_tag = []                   #存放单个句子词性标注的结果


        for j in i:
            if j in a1:
                flag_one.append('entity'+str(entity_count))              #实体标注
                flag_entity.append('entity' + str(entity_count))         #用于后续形成实体对
                entity_count += 1
            else:
                flag_one.append('O')
        flag_all.append(flag_one)

###############将一个句子中所有的实体提取出来##############
        for m in flag_entity:
            a = int(flag_one.index(m))
            b = i[a]
            entity_orign.append(b)

#############将一个句子中所有实体的位置提取出来###############
        for index, nums in enumerate(i):
            count = 0
            for p in entity_orign:
                if nums == p:
                    count += 1
                    if count == 1:
                        position_index.append(index)
                    else:
                        continue

        # for p in entity_orign:
        #     _index = int(i.index(p))
        #     position_index.append(_index)

###################词性标注#######################
        for char in i:
            posseg_list = pseg.cut(char)
            b = ' '.join('%s/%s' % (word, tag) for (word, tag) in posseg_list)
            Cx_one_tag.append(b)
        CX_all_tag.append(Cx_one_tag)

######################提取实体对########################
        for n in itertools.combinations(entity_orign, 2):
            entity_pair.append(list(n))
        entity_pairs.append(entity_pair)

        #################提取实体对位置###################
        for n in itertools.combinations(position_index, 2):
            position_index_pair.append(list(n))
        position_index_pairs.append(position_index_pair)

#####################复制实体对对应的句子######################
        copy_num = int((1 + (entity_count - 2)) * (entity_count - 2) / 2)
        for num in range(copy_num):
            sentence_copy.append(i)
        sentences_copy.append(sentence_copy)

##################将三维列表变成二维列表########################
    from itertools import chain
    entity_pairs = list(chain.from_iterable(entity_pairs))
    position_index_pairs = list(chain.from_iterable(position_index_pairs))
    sentences_copy = list(chain.from_iterable(sentences_copy))

#################将分词的句子恢复原状（将多列表元素变成单列表元素）##################
    for one_sen in sentences_copy:
        temp_sentence.append(''.join(one_sen))
        temp_sentences.append(temp_sentence)
        temp_sentence = []

###################按行合并###################
    un = []
    for i in range(len(entity_pairs)):
        un.append(['unknown', 'unknown', 'unknown'])


    d = np.hstack((entity_pairs, position_index_pairs)).tolist()                       ##将实体对和相应的位置合并
    g = np.hstack((d, un)).tolist()
    f = np.hstack((g, temp_sentences)).tolist()                                        ##将前者合并的结果与相应的句子合并



###################二维列表存入txt中#######################
    ######################二维列表写入文本####################
    for i in range(len(f)):
        for j in range(len(f[i])):
            outputfile.write(str(f[i][j]))
            outputfile.write(' ')
        outputfile.write('\n')
    outputfile.close()

    return f

def Relation_flag(file, outfile):
    ##############创建列表对读取的文件分列存储##############
    entity1 = []                ###存放读取的第一个实体词汇
    entity2 = []                ###存放读取的第二个实体词汇
    entity1_pos = []            ###存放第一个实体的位置索引
    entity2_pos = []            ###存放第二个实体的位置索引
    relationword = []           ###存放关系词汇
    relationword_pos = []       ###存放关系词汇的位置索引
    relationword_CX = []        ###存放关系词汇的词性
    sentences = []              ###保存原始句子
    CX = []                     ###用于关系词的词性标注
    Is_prep_or_conj = []        ###判断两个实体词汇之间是有介词或连词

    ############调用读取文件代码###########
    data = data_load2(file)
    print(data)

    for i in data:
        entity1.append(i[0])
        entity2.append(i[1])
        entity1_pos.append(i[2])
        entity2_pos.append(i[3])
        relationword.append(i[4])
        sentences.append(i[7])

    for relation_w in relationword:
        if relation_w == 'unknown':
            CX.append('unknown')
        else:
            posseg_list = pseg.cut(relation_w)
            b = ' '.join('%s/%s' % (word, tag) for (word, tag) in posseg_list)
            _index = b.index('/')               ###找到/的索引位置
            CX.append(b[_index+1:])
    relationword_CX = CX
    print(CX)

    #############读取分词的句子，用于提取关系词的位置#############
    count = 0
    for sentence in sentences:
        position_index = []
        count += 1
        line = trim(sentence)
        line_seg = segment_sentence(line)
        line_seg = line_seg.strip()
        sen = line_seg.split(' ')
        if relationword[count-1] == 'unknown':
            relationword_pos.append('unknown')
        else:
            word = relationword[count-1]
            start = int(entity1_pos[count - 1])
            a = sen.index(word, start)
            relationword_pos.append(a)
            # relationword_pos.append(sen.index(relationword[count-1]))
            # for index, nums in enumerate(sen):
            #     num_count =0
            #     if nums == relationword[count-1]:
            #         num_count += 1
            #         if num_count == 1:
            #             position_index.append(index)
            #         else:
            #             continue
            #
            # for i in position_index:
            #     a = int(entity1_pos[count - 1])
            #     b = int(entity2_pos[count - 1])
            #     if int(i) > a and int(i - b) < 9:  #######限制出现相同关系词时，保证关系词的位置在对应实体之后，最远距离实体2有5个词的距离
            #         relationword_pos.append(i)

    ##############判断存在关系词时，两个实体之间是否有介词或者连词，有用1标记，没有用0标记###############
    count_ = 0
    for sentence in sentences:
        is_prep_or_conj_1 = []
        count_ += 1
        line = trim(sentence)
        line_seg = segment_sentence(line)
        line_seg = line_seg.strip()
        sen = line_seg.split(' ')
        if relationword[count_ - 1] == 'unknown':
            Is_prep_or_conj.append('0')
        else:
            line = trim(sentence)
            line_seg = segment_sentence(line)
            line_seg = line_seg.strip()
            sen = line_seg.split(' ')
            for char in sen:
                posseg_list = pseg.cut(char)
                b = ' '.join('%s/%s' % (word, tag) for (word, tag) in posseg_list)
                is_prep_or_conj_1.append(b)


            end_entity_index = int(relationword_pos[count_ - 1])
            start_entity_index = end_entity_index-2                     ###判断关系词左边两个词的范围内是否有介词或者是连词
            is_prep_or_conj_2 = is_prep_or_conj_1[start_entity_index:end_entity_index]

            is_prep_or_conj_3 = []
            for i in is_prep_or_conj_2:
                _index = i.index('/')  ###找到/的索引位置
                is_prep_or_conj_3.append(i[_index + 1:])

            tongji = []
            for i in is_prep_or_conj_3:
                if i in ['p', 'c']:
                    tongji.append('1')

            if len(tongji) > 0:
                Is_prep_or_conj.append('1')
            else:
                Is_prep_or_conj.append('0')

    # relationword_pos.pop()
    entity1 = np.array(entity1).reshape(len(entity1), 1)  # reshape(列的长度，行的长度)
    entity2 = np.array(entity2).reshape(len(entity2), 1)  # reshape(列的长度，行的长度)
    entity1_pos = np.array(entity1_pos).reshape(len(entity1_pos), 1)  # reshape(列的长度，行的长度)
    entity2_pos = np.array(entity2_pos).reshape(len(entity2_pos), 1)  # reshape(列的长度，行的长度)
    relationword = np.array(relationword).reshape(len(relationword), 1)  # reshape(列的长度，行的长度)
    relationword_pos = np.array(relationword_pos).reshape(len(relationword_pos), 1)  # reshape(列的长度，行的长度)
    relationword_CX = np.array(relationword_CX).reshape(len(relationword_CX), 1)  # reshape(列的长度，行的长度)
    Is_prep_or_conj = np.array(Is_prep_or_conj).reshape(len(Is_prep_or_conj), 1)  # reshape(列的长度，行的长度)
    sentences = np.array(sentences).reshape(len(sentences), 1)

    ###################按行合并###################
    d = np.hstack((entity1, entity2)).tolist()  ##将实体对和相应的位置合并
    f = np.hstack((d, entity1_pos)).tolist()  ##将前者合并的结果与相应的句子合并
    g = np.hstack((f, entity2_pos)).tolist()
    h = np.hstack((g, relationword)).tolist()
    k = np.hstack((h, relationword_pos)).tolist()
    l = np.hstack((k, relationword_CX)).tolist()
    o = np.hstack((l, Is_prep_or_conj)).tolist()
    p = np.hstack((o, sentences)).tolist()

    ###################二维列表存入txt中#######################
    ######################二维列表写入文本####################
    for i in range(len(p)):
        for j in range(len(p[i])):
            outfile.write(str(p[i][j]))
            outfile.write(' ')
        outfile.write('\n')
    outfile.close()

    return p


if __name__ == '__main__':
    # inputs1 = open('result/train_data_pre_100.txt', 'r', encoding='utf-8')
    # sen = data_load1(inputs1)
    # inputs1.close()
    #
    # inputs2 = open('result/train_data_pre_100.txt', 'r', encoding='utf-8')
    # word = data_load2(inputs2)
    # data['word'] = word
    # inputs2.close()

    # data_load2(inputs2)

    # inputs1 = '关系提取关系词库.txt'
    # outputs1 = '关系提取关系词库_quchong.txt'
    # dic = quchong(inputs1, outputs1)

    # inputs1 = '关系词库.txt'
    # outputs1 = '去重关系词库.txt'
    # dic = quchong(inputs1, outputs1)

#################实体标注及实体位置标注##################
    inputs = open('result/train_data_pre_100_all.txt', 'r', encoding='utf-8')
    outputs = open('result/实体对及位置提取结果_for_all-753.txt', 'w', encoding='utf-8')
    entitycankaoku = '关系提取实体库_去重_可扩展.txt'
    relationku = '关系提取关系词库_quchong.txt'
    biaozhu_entity(entitycankaoku, inputs,outputs)

# ################关系词词性标注及位置标注##################
#     inputs = open('result/实体对及位置提取结果_for_all-756.txt', 'r', encoding='utf-8')
#     outputs = open('result/标注完成最终结果_for_all.txt', 'w', encoding='utf-8')
#     Relation_flag(inputs, outputs)


