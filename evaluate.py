#######用于关系词提取结果评估##########
"""
通过前面提取的特征，计算标注好的句子中每个词的权值，进而选择出概率最高的那个值作为关键词
"""
from Biaozhu import data_load2,trim,segment_sentence
import jieba.posseg as pseg
import numpy as np

#求列表最大值的位置索引
def max_fun_index(lst_int):
    index = []
    max_n = max(lst_int)
    for i in range(len(lst_int)):
        if lst_int[i] == max_n:
            index.append(i)
    return index

# 加载实体词
def getwords(filepath):
    stopwords = [line.strip()
        for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def evaluate_score(file):
    ########特征录入########
    POS = [0.827, 0.147, 0.013, 0.013]
    LOC = [0.75, 0.366, 0.25, 0.634, 0]
    DIS_entity = [0.237, 0.113, 0.013, 0.236, 0.058, 0.240, 0.103, 0]
    DIS_punc = [0.205, 0.084, 0.306, 0.275, 0.13, 0]
    noun_len = [1, 2, 3, 4]
    verb_len = [1, 2]
    prep_len = [1]
    other_len = [3]

    ##############创建列表对读取的文件分列存储##############
    entity1 = []  ###存放读取的第一个实体词汇
    entity2 = []  ###存放读取的第二个实体词汇
    entity1_pos = []  ###存放第一个实体的位置索引
    entity2_pos = []  ###存放第二个实体的位置索引
    relationword = []  ###存放关系词汇
    relationword_pos = []  ###存放关系词汇的位置索引
    relationword_CX = []  ###存放关系词汇的词性
    sentences = []  ###保存原始句子
    Is_prep_or_conj = []  ###判断两个实体词汇之间是有介词或连词

    ########计算得分########
    data1 = data_load2(file)
    #########抽样#########
    batch = []
    for i in range(1000):
        batch.append(np.random.choice(len(data1), 10000))
    #
    # #########将抽样的数字还原成标注完成的句子#########
    batches = []
    for j in batch:
        yuanwen_batch = []
        for k in j:
            yuanwen_batch.append(data1[k])
        batches.append(yuanwen_batch)
    # batches.append(data1)

    mean_accuracy = []
    mean_precision = []
    mean_recall = []
    mean_F1 = []
    for data in batches:
        for i in data:
            entity1.append(i[0])
            entity2.append(i[1])
            entity1_pos.append(i[2])
            entity2_pos.append(i[3])
            relationword.append(i[4])
            relationword_pos.append(i[5])
            relationword_CX.append(i[6])
            Is_prep_or_conj.append(i[7])
            sentences.append(i[8])

        count = 0
        word_sentence = []
        word_CX_score_sentence = []
        word_LOC_score_sentence = []
        word_DIS_entity_score_sentence = []
        word_DIS_punc_score_sentence = []
        sens = []
        entity1_slice_indexs = []
        entity2_slice_indexs = []
        for sentence in sentences:
            count += 1
            line = trim(sentence)
            line_seg = segment_sentence(line)
            line_seg = line_seg.strip()
            sen = line_seg.split(' ')
            sens.append(sen)

            # 计算句子中每个词的词性
            word_CX = []
            for i in sen:
                posseg_list = pseg.cut(i)
                b = ' '.join('%s/%s' % (word, tag) for (word, tag) in posseg_list)
                _index = b.index('/')  ###找到/的索引位置
                cx = b[_index + 1:]
                word_CX.append(cx)

            # 标记句子中所有逗号和句号所处的位置
            BiaoDian_index = []
            for index, nums in enumerate(sen):
                a_count = 0
                for p in ['，', '。', '；']:
                    if nums == p:
                        a_count += 1
                        if a_count == 1:
                            BiaoDian_index.append(index)
                        else:
                            continue

            entity1_slice_index = []
            # 寻找距离实体一最近的两个标点的位置
            entity1_index = int(entity1_pos[count - 1])
            XD_dis = []
            if len(BiaoDian_index) <= 1:
                before_index1 = -1
                back_index1 = BiaoDian_index[0]
            else:
                for k in BiaoDian_index:
                    xiangdui = k - entity1_index
                    XD_dis.append(xiangdui)

                # 找到实体一后面第一个逗号、分号或者句号所处的位置
                temp1 = []
                for k in XD_dis:
                    if int(k) < 0:
                        temp1.append(999)
                    else:
                        temp1.append(k)
                min_index = temp1.index(int(min(temp1)))
                back_index1 = BiaoDian_index[min_index]

                # 找到实体一前面第一个逗号、分号或者句号所处的位置
                temp2 = []
                for k in XD_dis:
                    if int(k) > 0:
                        temp2.append(-999)
                    else:
                        temp2.append(k)

                if len(set(temp2)) == 1:  # 判断是不是全是-999，如果是的话直接取-1为前一个标点位置的索引
                    before_index1 = -1
                else:
                    max_index = temp2.index(int(max(temp2)))
                    before_index1 = BiaoDian_index[max_index]

            entity1_slice_index.append(before_index1)
            entity1_slice_index.append(back_index1)
            entity1_slice_indexs.append(entity1_slice_index)

            entity2_slice_index = []
            # 寻找距离实体二最近的两个标点的位置
            entity2_index = int(entity2_pos[count - 1])
            XD_dis = []
            if len(BiaoDian_index) <= 1:
                before_index = -1
                back_index = BiaoDian_index[0]
            else:
                for k in BiaoDian_index:
                    xiangdui = k - entity2_index
                    XD_dis.append(xiangdui)

                # 找到实体二后面第一个逗号、分号或者句号所处的位置
                temp1 = []
                for k in XD_dis:
                    if int(k) < 0:
                        temp1.append(999)
                    else:
                        temp1.append(k)
                min_index = temp1.index(int(min(temp1)))
                back_index = BiaoDian_index[min_index]

                # 找到实体二前面第一个逗号、分号或者句号所处的位置
                temp2 = []
                for k in XD_dis:
                    if int(k) > 0:
                        temp2.append(-999)
                    else:
                        temp2.append(k)

                if len(set(temp2)) == 1:  # 判断是不是全是-999，如果是的话直接取-1为前一个标点位置的索引
                    before_index = -1
                else:
                    max_index = temp2.index(int(max(temp2)))
                    before_index = BiaoDian_index[max_index]

            entity2_slice_index.append(before_index)
            entity2_slice_index.append(back_index)
            entity2_slice_indexs.append(entity2_slice_index)

            word = []
            word_CX_score = []
            word_LOC_score = []
            word_DIS_entity_score = []
            word_DIS_punc_score = []

            i_index = -1
            for i in sen:
                i_index += 1
                word.append(i)

                # 计算每个词的词性得分
                posseg_list = pseg.cut(i)
                b = ' '.join('%s/%s' % (word, tag) for (word, tag) in posseg_list)
                _index = b.index('/')  ###找到/的索引位置
                cx = b[_index + 1:]
                if cx in ['n', 'f', 'ns', 'nt', 'nr']:
                    word_CX_score.append(POS[0])
                elif cx == 'v':
                    word_CX_score.append(POS[1])
                elif cx == 'p':
                    word_CX_score.append(POS[2])
                else:
                    word_CX_score.append(POS[3])
                word_CX.append(cx)  # 记录单个句子每个词的词性

                entity1_pos_index = int(entity1_pos[count - 1])  # 该句子中实体1的位置
                entity2_pos_index = int(entity2_pos[count - 1])  # 该句子中实体2的位置
                word_index = i_index  # 获取每个词的索引位置

                # 计算每个词的位置得分
                a = 0
                # word_p_or_c_split = []
                if word_index >= 3:
                    word_p_or_c_split = word_CX[word_index - 3:word_index]
                    for i in word_p_or_c_split:
                        if i in ['p', 'c']:
                            a = 1
                        else:
                            continue

                if word_index > entity2_pos_index and a == 1:
                    word_LOC_score.append(LOC[0])
                elif word_index > entity2_pos_index:
                    word_LOC_score.append(LOC[1])
                elif entity1_pos_index < word_index < entity2_pos_index and a == 1:
                    word_LOC_score.append(LOC[2])
                elif entity1_pos_index < word_index < entity2_pos_index:
                    word_LOC_score.append(LOC[3])
                else:
                    word_LOC_score.append(LOC[4])

                # 计算每个词距离实体词的距离得分
                if word_index > entity2_pos_index and word_index - entity2_pos_index == 1:
                    word_DIS_entity_score.append(DIS_entity[0])
                elif word_index > entity2_pos_index and 1 < word_index - entity2_pos_index < 5:
                    word_DIS_entity_score.append(DIS_entity[1])
                elif word_index > entity2_pos_index and word_index - entity2_pos_index >= 5:
                    word_DIS_entity_score.append(DIS_entity[2])
                elif word_index < entity2_pos_index and 1 <= word_index - entity1_pos_index < 5:
                    word_DIS_entity_score.append(DIS_entity[3])
                elif word_index < entity2_pos_index and 5 <= word_index - entity1_pos_index < entity2_pos_index - entity1_pos_index - 4:
                    word_DIS_entity_score.append(DIS_entity[4])
                elif word_index < entity2_pos_index and 1 <= entity2_pos_index - word_index <= 2:
                    word_DIS_entity_score.append(DIS_entity[5])
                elif word_index < entity2_pos_index and 2 < entity2_pos_index - word_index <= 4:
                    word_DIS_entity_score.append(DIS_entity[6])
                else:
                    word_DIS_entity_score.append(DIS_entity[7])

                # 计算每个词距离实体二前后标点符号的距离
                if back_index - word_index == 1:
                    word_DIS_punc_score.append(DIS_punc[0])
                elif word_index - before_index == 1:
                    word_DIS_punc_score.append(DIS_punc[2])
                elif 1 < back_index - word_index < 5:
                    word_DIS_punc_score.append(DIS_punc[1])
                elif 1 < word_index - before_index < 5:
                    word_DIS_punc_score.append(DIS_punc[3])
                elif before_index + 5 <= word_index <= back_index - 5:
                    word_DIS_punc_score.append(DIS_punc[4])
                else:
                    word_DIS_punc_score.append(DIS_punc[5])

            word_sentence.append(word)
            word_CX_score_sentence.append(word_CX_score)
            word_LOC_score_sentence.append(word_LOC_score)
            word_DIS_entity_score_sentence.append(word_DIS_entity_score)
            word_DIS_punc_score_sentence.append(word_DIS_punc_score)

        # 计算每个词的得分
        socre_sentences = []  ###存放所有句子每个词的权重得分
        for i in range(len(word_sentence)):
            socre_sentences.append(np.sum(
                [word_CX_score_sentence[i], word_LOC_score_sentence[i], word_DIS_entity_score_sentence[i]], axis=0) / 3)
        # , word_DIS_punc_score_sentence[i]
        # 筛选关系词
        relationwords = []
        relationwords_index = []
        slice_sentence_ = []
        count_ = 0
        for i in socre_sentences:
            count_ += 1
            slice_index_entity2 = entity2_slice_indexs[count_ - 1]
            list1 = i.tolist()
            slice_sentence = list1[slice_index_entity2[0] + 1:slice_index_entity2[1]]
            if len(slice_sentence) > 0:
                a = max(slice_sentence)
                if a <= 0.34:                             ###阈值调整
                    relationwords_index.append(['unknown'])
                    continue
                else:
                    slice_sentence_ = max_fun_index(slice_sentence)
                    c = [l + slice_index_entity2[0] + 1 for l in slice_sentence_]
                    relationwords_index.append(c)
            else:
                relationwords_index.append(['unknown'])

        count_num = -1
        for i in relationwords_index:
            relationwords_temp = []
            count_num += 1
            for j in i:
                if j == 'unknown':
                    relationwords_temp.append(j)
                else:
                    relationwords_temp.append(sens[count_num][j])
            relationwords.append(relationwords_temp)

        # # 根据关系词长度进行最终筛选
        entitys = getwords('./关系提取实体库.txt')
        # stopwords = getwords('./result/stopwords_去除地质实体关系词.txt')
        relationwords_2 = []
        for i in relationwords:
            relationwords_temp2 = []
            for j in i:
                if j == 'unknown':
                    continue
                if j in entitys:
                    continue
                # elif j in stopwords:
                #     continue
                else:
                    # relationwords_temp2.append(j)
                    posseg_list = pseg.cut(j)
                    b = ' '.join('%s/%s' % (word, tag) for (word, tag) in posseg_list)
                    _index = b.index('/')  ###找到/的索引位置
                    cx = b[_index + 1:]
                    if cx in ['n', 'f', 'ns', 'nt', 'nr']:
                        if 1 <= len(j) <= 4:
                            relationwords_temp2.append(j)
                        else:
                            continue
                    elif cx == 'v':
                        if 1 <= len(j) <= 2:
                            relationwords_temp2.append(j)
                        else:
                            continue
                    elif cx == 'p':
                        if len(j) == 1:
                            relationwords_temp2.append(j)
                        else:
                            continue
                    else:
                        if 2 <= len(j) <= 3:
                            relationwords_temp2.append(j)
                        else:
                            continue
            if len(relationwords_temp2) == 0:
                relationwords_temp2.append('unknown')
            relationwords_2.append(relationwords_temp2)

        relationwords_all = getwords('./关系提取关系词库.txt')
        relationwords_3 = []
        for i in relationwords_2:
            relationwords_temp3 = []
            for j in i:
                if j in relationwords_all:
                    relationwords_temp3.append(j)
            if len(relationwords_temp3) == 0:
                relationwords_temp3.append('unknown')
            relationwords_3.append(relationwords_temp3)

        # 构建三元组
        Triads = []
        for i in range(len(socre_sentences)):
            Triad = []
            Triad.append(entity1[i])
            Triad.append(entity2[i])
            slice_index_entity1 = entity1_slice_indexs[i]
            slice_index_entity2 = entity2_slice_indexs[i]

            if slice_index_entity1 != slice_index_entity2:
                if slice_index_entity1[0] == -1:
                    entity1_index = int(entity1_pos[i])
                    sentence_1 = sentences[i]
                    line = trim(sentence_1)
                    line_seg = segment_sentence(line)
                    line_seg = line_seg.strip()
                    sen = line_seg.split(' ')
                    index_temp = []
                    for m in sen:
                        if m in entitys:
                            index_temp.append(sen.index(m))
                    if entity1_index != index_temp[0]:
                        Triad.append('unknown')
                    else:
                        entity2_index = int(entity2_pos[i])
                        # 筛选距离实体2位置最近的词作为关系词
                        if len(relationwords_3[i]) > 1:
                            temp = []
                            for j in relationwords_3[i]:
                                temp.append(entity2_index - sens[i].index(j))
                            temp1 = []
                            for k in temp:
                                if int(k) < 0:
                                    temp1.append(999)
                                else:
                                    temp1.append(k)

                            min_index = temp1.index(int(min(temp1)))
                            Triad.append(relationwords_3[i][min_index])
                        else:
                            Triad.append(relationwords_3[i][0])
                elif slice_index_entity1[0] != -1:
                    Triad.append('unknown')
                else:
                    entity2_index = int(entity2_pos[i])
                    # 筛选距离实体2位置最近的词作为关系词
                    if len(relationwords_3[i]) > 1:
                        temp = []
                        for j in relationwords_3[i]:
                            temp.append(entity2_index - sens[i].index(j))
                        temp1 = []
                        for k in temp:
                            if int(k) < 0:
                                temp1.append(999)
                            else:
                                temp1.append(k)
                        min_index = temp1.index(int(min(temp1)))
                        Triad.append(relationwords_3[i][min_index])
                    else:
                        Triad.append(relationwords_3[i][0])
            else:
                entity2_index = int(entity2_pos[i])
                # 筛选距离实体2位置最近的词作为关系词
                if len(relationwords_3[i]) > 1:
                    temp = []
                    for j in relationwords_3[i]:
                        temp.append(entity2_index - sens[i].index(j))
                    temp1 = []
                    for k in temp:
                        if int(k) < 0:
                            temp1.append(999)
                        else:
                            temp1.append(k)

                    min_index = temp1.index(int(min(temp1)))
                    Triad.append(relationwords_3[i][min_index])
                else:
                    Triad.append(relationwords_3[i][0])
            Triads.append(Triad)

        count_triple = -1
        for i in Triads:
            count_triple += 1
            i.append(sentences[count_triple])

#         Triads_out = []
#         for i in Triads:
#             if i[2] != 'unknown':
#                 Triads_out.append(i)
        
#         for i in range(len(Triads_out)):
#             for j in range(len(Triads_out[i])):
#                 outfile.write(str(Triads_out[i][j]))
#                 outfile.write(' ')
#             outfile.write('\n')
#         outfile.close()


        ##########计算准确率###########
        predict = [i[2] for i in Triads]
        really = relationword

        TP_count = 0  # 真正例
        TN_count = 0  # 真负例
        FN_count = 0  # 假负例
        FP_count = 0  # 假正例

        for i in range(len(predict)):
            if predict[i] == really[i] and really[i] != 'unknown':
                TP_count += 1
            elif predict[i] != really[i] and really[i] != 'unknown':
                FN_count += 1
            elif predict[i] == really[i] and really[i] == 'unknown':
                TN_count += 1
            elif predict[i] != really[i] and really[i] == 'unknown':
                FP_count += 1

        # 计算准确率召回率
        # precision = 所有识别正确的关系词/所有被识别为关系的词个数
        # recall = 所有识别正确的关系词个数/被标识为关系词的个数
        Accuracy = (TP_count + TN_count) / (TP_count + TN_count + FN_count + FP_count)
        precision = TP_count / (TP_count + FP_count)
        recall = TP_count / (TP_count + FN_count)
        F1 = 2 * precision * recall / (precision + recall)
        mean_accuracy.append(Accuracy)
        mean_precision.append(precision)
        mean_recall.append(recall)
        mean_F1.append(F1)

        # print('Acc:%s Precision:%s Recall:%s F1_score:%s' % (Accuracy, precision, recall, F1))

    mean_Accuracy = np.mean(mean_accuracy)
    mean_Precision = np.mean(mean_precision)
    mean_Recall = np.mean(mean_recall)
    mean_F1_score = np.mean(mean_F1)
    print('Acc:%s Precision:%s Recall:%s F1_score:%s' % (mean_Accuracy, mean_Precision, mean_Recall, mean_F1_score))


    return 0         #######返回准确率#######



if __name__ == '__main__':
    inputs = open('result/train_data_pre_100_test_biaozhu_new_最终结果.txt', 'r', encoding='utf-8')

    # inputs = open('没有识别的句子.txt', 'r', encoding='utf-8')
    ####提取所有的三元组信息####
    # inputs = open('result/实体对及位置提取结果_for_all-753.txt', 'r', encoding='utf-8')
    # outputs = open('result/triples_of_all.txt', 'w', encoding='utf-8')
    # outputs = open('result/triples_of_all_haverelation.txt', 'w', encoding='utf-8')
    evaluate_score(inputs)
