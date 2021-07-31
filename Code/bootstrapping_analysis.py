"""
代码作用：利用bootstrapping算法进行特征统计
本节代码的作用是提取Bootstrapping各项特征系数
本实验选择的特征包括（1)词性特征 （2）关系词所处的位置 （3）句法依存关系 （4）左边有连词或者介词时的位置 （5）到e1的距离 （6）到e2的距离 （7）关系词的长度 （8）e1到e2的距离
"""
import numpy as np
from label import data_load2
import jieba.posseg as pseg
from dataprocess import trim, segment_sentence
import matplotlib
import matplotlib.pyplot as plt

def Bootstrapping(file):
    yuanwen = data_load2(file)

    #########抽样#########
    batch = []
    for i in range(10000):
        batch.append(np.random.choice(len(yuanwen), 1000))


    #########将抽样的数字还原成标注完成的句子#########
    batches = []
    for j in batch:
        yuanwen_batch = []
        for k in j:
            yuanwen_batch.append(yuanwen[k])
        batches.append(yuanwen_batch)
    
    # 统计关系词位于实体二所处句子切片的位置
    Position = []
    for i in batches:
        position = []
        left_1_num = 0      #关系词位于最左边的数量
        left_2_num = 0      #关系词位于最左边第二个位置
        # left_3_num = 0      #关系词位于最左边第三个位置
        # left_4_num = 0  # 关系词位于最左边第四个位置
        between_num = 0     #关系词位于最左边第五个位置到倒数第五个位置
        right_1_num = 0     #关系词位于最右边的数量
        right_2_num = 0     #关系词位于最右边倒数第二个位置
        # right_3_num = 0     #关系词位于最右边倒数第三个位置
        # right_4_num = 0  # 关系词位于最右边倒数第四个位置
    
        between_num = 0   #关系词位于中间的数量
        for j in i:
            sentence = j[8]
            line = trim(sentence)
            line_seg = segment_sentence(line)
            line_seg = line_seg.strip()
            sen = line_seg.split(' ')
    
            #标记句子中所有逗号和句号所处的位置
            BiaoDian_index = []
            for index, nums in enumerate(sen):
                count = 0
                for p in ['，', '。', '；']:
                    if nums == p:
                        count += 1
                        if count == 1:
                            BiaoDian_index.append(index)
                        else:
                            continue
    
            #寻找距离实体二最近的两个标点的位置
            entity2_index = int(j[3])
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
    
                if len(set(temp2)) == 1:        #判断是不是全是-999，如果是的话直接取-1为前一个标点位置的索引
                    before_index = -1
                else:
                    max_index = temp2.index(int(max(temp2)))
                    before_index = BiaoDian_index[max_index]
    
    
            if j[5] == 'unknown':
                continue
            elif (int(j[5]) - before_index) == 1:  # 判断关系词是不是前面一个逗号或者句号的下一位
                left_1_num += 1
                continue
            elif (int(j[5]) - before_index) == 2:
                left_2_num += 1
                continue
            # elif (int(j[5]) - before_index) == 3:
            #     left_3_num += 1
            #     continue
            # elif (int(j[5]) - before_index) == 4:
            #     left_4_num += 1
            #     continue
            elif (back_index - int(j[5])) == 1:
                right_1_num += 1
                continue
            elif (back_index - int(j[5])) == 2:
                right_2_num += 1
                continue
            # elif (back_index - int(j[5])) == 3:
            #     right_3_num += 1
            #     continue
            # elif (back_index - int(j[5])) == 4:
            #     right_4_num += 1
            #     continue
            elif before_index + 2 < int(j[5]) < back_index - 2:
                between_num += 1
                continue
        position.append(left_1_num)
        position.append(left_2_num)
        # position.append(left_3_num)
        # position.append(left_4_num)
        position.append(between_num)
        # position.append(right_4_num)
        # position.append(right_3_num)
        position.append(right_2_num)
        position.append(right_1_num)
        Position.append(position)
    
    left_1_num_percent_all = []
    left_2_num_percent_all = []
    left_3_num_percent_all = []
    left_4_num_percent_all = []
    between_num_percent_all = []
    right_4_num_percent_all = []
    right_3_num_percent_all = []
    right_2_num_percent_all = []
    right_1_num_percent_all = []
    
    for i in Position:
        left_1_num_percent = i[0]/(i[0]+i[1]+i[2]+i[3]+i[4]+i[5]+i[6]+i[7]+i[8])
        left_2_num_percent = i[1] / (i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6] + i[7] + i[8])
        left_3_num_percent = i[2] / (i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6] + i[7] + i[8])
        left_4_num_percent = i[3] / (i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6] + i[7] + i[8])
        between_num_percent = i[4] / (i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6] + i[7] + i[8])
        right_4_num_percent = i[5] / (i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6] + i[7] + i[8])
        right_3_num_percent = i[6] / (i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6] + i[7] + i[8])
        right_2_num_percent = i[7] / (i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6] + i[7] + i[8])
        right_1_num_percent = i[8] / (i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6] + i[7] + i[8])
    
        left_1_num_percent_all.append(left_1_num_percent)
        left_2_num_percent_all.append(left_2_num_percent)
        left_3_num_percent_all.append(left_3_num_percent)
        left_4_num_percent_all.append(left_4_num_percent)
        between_num_percent_all.append(between_num_percent)
        right_4_num_percent_all.append(right_4_num_percent)
        right_3_num_percent_all.append(right_3_num_percent)
        right_2_num_percent_all.append(right_2_num_percent)
        right_1_num_percent_all.append(right_1_num_percent)
    
    left_1_num_average = np.mean(left_1_num_percent_all)
    left_2_num_average = np.mean(left_2_num_percent_all)
    left_3_num_average = np.mean(left_3_num_percent_all)
    left_4_num_average = np.mean(left_4_num_percent_all)
    between_num_average = np.mean(between_num_percent_all)
    right_4_num_average = np.mean(right_4_num_percent_all)
    right_3_num_average = np.mean(right_3_num_percent_all)
    right_2_num_average = np.mean(right_2_num_percent_all)
    right_1_num_average = np.mean(right_1_num_percent_all)
    
    labels2 = ['Left_one', 'Left_two', 'Left_three', 'Left_four', 'Middle',
               'Right_four', 'Right_three', 'Right_two', 'Right_one']
    # 绘图显示的标签
    values2 = [left_1_num_average, left_2_num_average, left_3_num_average, left_4_num_average,
               between_num_average, right_4_num_average, right_3_num_average, right_2_num_average, right_1_num_average]
    colors2 = [plt.cm.Accent(i) for i in np.linspace(0, 1, len(labels2))]
    # 将排列在第4位的语言(Python)分离出来
    explode2 = [0, 0.01, 0.02, 0.04, 0.06, 0.04, 0.02, 0.01, 0]
    # 旋转角度
    plt.title("Position of relative words in sentence slice(including entity 2)", fontsize=20)
    # 标题
    plt.pie(values2, labels=labels2, explode=explode2, colors=colors2, pctdistance=0.9,
            startangle=180,
            shadow=True, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()


    ##########词性统计##########
    CX = []
    for i in batches:
        cx = []
        unknown_num = 0
        noun_num = 0
        prep_num = 0
        verb_num = 0
        other_num = 0
        for j in i:
            if j[6] == 'unknown':
                unknown_num += 1
            elif j[6] == 'v':
                verb_num += 1
            elif j[6] == 'p':
                prep_num += 1
            elif j[6] == 'f':
                noun_num += 1
            elif j[6] == 'ns':
                noun_num += 1
            elif j[6] == 'n':
                noun_num += 1
            elif j[6] == 'nr':
                noun_num += 1
            elif j[6] == 'nt':
                noun_num += 1
            else:
                other_num += 1
                print(j[4])
        cx.append(unknown_num)
        cx.append(noun_num)
        cx.append(verb_num)
        cx.append(prep_num)
        cx.append(other_num)
        CX.append(cx)
    
    unknown_num_percent_all = []
    noun_num_percent_all = []
    verb_num_percent_all = []
    prep_num_percent_all = []
    other_num_percent_all = []
    
    for i in CX:
        unknown_num_percent = i[0]/1000
        noun_num_percent = i[1]/1000
        verb_num_percent = i[2]/1000
        prep_num_percent = i[3]/1000
        other_num_percent = i[4]/1000
        unknown_num_percent_all.append(unknown_num_percent)
        noun_num_percent_all.append(noun_num_percent)
        verb_num_percent_all.append(verb_num_percent)
        prep_num_percent_all.append(prep_num_percent)
        other_num_percent_all.append(other_num_percent)
    
    unknown_num_average = np.mean(unknown_num_percent_all)
    noun_num_average = np.mean(noun_num_percent_all)
    verb_num_average = np.mean(verb_num_percent_all)
    prep_num_average = np.mean(prep_num_percent_all)
    other_num_average = np.mean(other_num_percent_all)
    print(unknown_num_average)
    
    noun_num_average_1 = noun_num_average/(noun_num_average + verb_num_average + prep_num_average + other_num_average)
    verb_num_average_1 = verb_num_average/(noun_num_average + verb_num_average + prep_num_average + other_num_average)
    prep_num_average_1 = prep_num_average/(noun_num_average + verb_num_average + prep_num_average + other_num_average)
    other_num_average_1 = other_num_average/(noun_num_average + verb_num_average + prep_num_average + other_num_average)
    
    labels1 = ['noun', 'verb', 'prep', 'other']
    # 绘图显示的标签
    values1 = [noun_num_average_1, verb_num_average_1, prep_num_average_1, other_num_average_1]
    # colors = ['y', 'm', 'b']
    colors1 = [plt.cm.Accent(i) for i in np.linspace(0, 1, 4)]
    # 将排列在第4位的语言(Python)分离出来
    explode1 = [0.03, 0.02, 0.01, 0]
    # 旋转角度
    plt.title("Part of speech proportion diagram", fontsize=20)
    # 标题
    plt.pie(values1, labels=labels1, explode=explode1, colors=colors1,
            startangle=180,
            shadow=True, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()

    #########关系词位置统计(在e1左边，e2右边，还是e1,e2中间)#########
    POS = []
    for i in batches:
       pos = []
       left_num = 0
       right_num = 0
       between_num = 0
       for j in i:
           if j[5] == 'unknown':
               continue
           elif int(j[5]) < int(j[2]):
               left_num += 1
               print(j)
           elif int(j[2])<int(j[5])<int(j[3]):
               between_num += 1
           else:
               right_num += 1
       pos.append(left_num)
       pos.append(between_num)
       pos.append(right_num)
       POS.append(pos)
    
    left_num_percent_all = []
    between_num_percent_all = []
    right_num_percent_all = []
    for i in POS:
        left_num_percent = i[0]/(i[0]+i[1]+i[2])
        between_num_percent = i[1]/(i[0]+i[1]+i[2])
        rigeht_num_percent = i[2]/(i[0]+i[1]+i[2])
        left_num_percent_all.append(left_num_percent)
        between_num_percent_all.append(between_num_percent)
        right_num_percent_all.append(rigeht_num_percent)
    left_num_average = np.mean(left_num_percent_all)
    between_num_average = np.mean(between_num_percent_all)
    right_num_average = np.mean(right_num_percent_all)
    
    labels2 = ['Left', 'Middle', 'Right']
    # 绘图显示的标签
    values2 = [left_num_average, between_num_average, right_num_average]
    colors2 = [plt.cm.Accent(i) for i in np.linspace(0, 1, len(labels2))]
    # 将排列在第4位的语言(Python)分离出来
    explode2 = [0, 0, 0.01]
    # 旋转角度
    plt.title("Relative word position", fontsize=20)
    # 标题
    plt.pie(values2, labels=labels2, explode=explode2, colors=colors2,
            startangle=180,
            shadow=True, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()

    #########关系词左边2个词中有连词或者介词时的位置(在e1左边，e2右边，还是e1,e2中间)#########
    Conditation_POS = []
    for i in batches:
        conditation_POS = []
        left_num = 0
        right_num = 0
        between_num = 0
        for j in i:
            if j[7] == '1':
                if int(j[5]) < int(j[2]):
                    left_num += 1
                elif int(j[2]) < int(j[5]) < int(j[3]):
                    between_num += 1
                else:
                    right_num += 1
            else:
                continue
        conditation_POS.append(left_num)
        conditation_POS.append(between_num)
        conditation_POS.append(right_num)
        Conditation_POS.append(conditation_POS)

    left_num_percent_all2 = []
    between_num_percent_all2 = []
    right_num_percent_all2 = []
    for i in Conditation_POS:
        left_num_percent = i[0] / (i[0] + i[1] + i[2])
        between_num_percent = i[1] / (i[0] + i[1] + i[2])
        rigeht_num_percent = i[2] / (i[0] + i[1] + i[2])
        left_num_percent_all2.append(left_num_percent)
        between_num_percent_all2.append(between_num_percent)
        right_num_percent_all2.append(rigeht_num_percent)
    left_num_average1 = np.mean(left_num_percent_all2)
    between_num_average1 = np.mean(between_num_percent_all2)
    right_num_average1 = np.mean(right_num_percent_all2)

    labels3 = ['Left', 'Middle', 'Right']
    # 绘图显示的标签
    values3 = [left_num_average1, between_num_average1, right_num_average1]
    colors3 = [plt.cm.Accent(i) for i in np.linspace(0, 1, len(labels3))]
    # 将排列在第4位的语言(Python)分离出来
    explode3 = [0, 0, 0.01]
    # 旋转角度
    plt.title("Relative word position (Left)", fontsize=20)
    # 标题
    plt.pie(values3, labels=labels3, explode=explode3, colors=colors3,
            startangle=180,
            shadow=True, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()

    #########关系词到e1的距离##########
    Dis1 = []
    for i in batches:
        dis1 = []
        for j in i:
            if j[5] == 'unknown':
                continue
            distance1 = (int(j[5]) - int(j[2]))
            dis1.append(distance1)
        Dis1.append(dis1)
    
    one_word_all = []
    two_words_all = []
    three_words_all = []
    four_words_all = []
    over_four_words_all = []
    
    for i in Dis1:
    
        one_word = i.count(1)/len(i)
        two_words = i.count(2)/len(i)
        three_words = i.count(3)/len(i)
        four_words = i.count(4)/len(i)
        over_words = (len(i) - i.count(1) - i.count(2) - i.count(3) - i.count(4))/len(i)
        one_word_all.append(one_word)
        two_words_all.append(two_words)
        three_words_all.append(three_words)
        four_words_all.append(four_words)
        over_four_words_all.append(over_words)
    
    one_wordaverage = np.mean(one_word_all)
    two_words_average = np.mean(two_words_all)
    three_words_average = np.mean(three_words_all)
    four_words_average = np.mean(four_words_all)
    over_words_average = np.mean(over_four_words_all)
    
    labels4 = ['One', 'Two', 'Three', 'Four', 'Above']
    # 绘图显示的标签
    values4 = [one_wordaverage, two_words_average, three_words_average, four_words_average, over_words_average]
    colors4 = [plt.cm.Accent(i) for i in np.linspace(0, 1, len(labels4))]
    # 将排列在第4位的语言(Python)分离出来
    explode4 = [0.02, 0.04, 0.06, 0.1, 0]
    # 旋转角度
    plt.title("The distance from the relationship word to E1", fontsize=20)
    # 标题
    plt.pie(values4, labels=labels4, explode=explode4, colors=colors4,
            startangle=180,
            shadow=True, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()


    #########关系词到e2的距离###########
    Dis2 = []
    for i in batches:
        dis2 = []
        for j in i:
            if j[5] == 'unknown':
                continue
            distance2 = (int(j[5]) - int(j[3]))
            dis2.append(distance2)
        Dis2.append(dis2)
    
    negative_others_all = []
    negative_one_word_all = []
    negative_two_words_all = []
    negative_three_words_all = []
    negative_four_words_all = []
    one_word_all = []
    two_words_all = []
    three_words_all = []
    four_words_all = []
    over_four_words_all = []
    
    for i in Dis2:
        a = len([j for j in i if j < 0])
        negative_one_words = i.count(-1)/len(i)
        negative_two_words = i.count(-2)/len(i)
        negative_three_words = i.count(-3)/len(i)
        negative_four_words = i.count(-4)/len(i)
        negative_others = (len([j for j in i if j < 0]) - i.count(-1) - i.count(-2) - i.count(-3) - i.count(-4)) / len(i)
        one_word = i.count(1) / len(i)
        two_words = i.count(2) / len(i)
        three_words = i.count(3) / len(i)
        four_words = i.count(4) / len(i)
        over_words = (len(i) - i.count(1) - i.count(2) - i.count(3) - i.count(4) - a) / len(i)
    
        negative_one_word_all.append(negative_one_words)
        negative_two_words_all.append(negative_two_words)
        negative_three_words_all.append(negative_three_words)
        negative_four_words_all.append(negative_four_words)
        negative_others_all.append(negative_others)
        one_word_all.append(one_word)
        two_words_all.append(two_words)
        three_words_all.append(three_words)
        four_words_all.append(four_words)
        over_four_words_all.append(over_words)
    
    negative_one_word_average = np.mean(negative_one_word_all)
    negative_two_words_average = np.mean(negative_two_words_all)
    negative_three_words_average = np.mean(negative_three_words_all)
    negative_four_words_average = np.mean(negative_four_words_all)
    negative_others_average = np.mean(negative_others_all)
    one_word_average = np.mean(one_word_all)
    two_words_average = np.mean(two_words_all)
    three_words_average = np.mean(three_words_all)
    four_words_average = np.mean(four_words_all)
    over_words_average = np.mean(over_four_words_all)
    
    labels4 = ['negative_Others', 'negative_Four', 'negative_Three', 'negative_Two', 'negative_One', 'One', 'Two', 'Three', 'Four', 'Above']
    # 绘图显示的标签
    values4 = [negative_others_average,negative_four_words_average,negative_three_words_average,negative_two_words_average,negative_one_word_average,
               one_word_average, two_words_average, three_words_average, four_words_average, over_words_average]
    
    colors4 = [plt.cm.Accent(i) for i in np.linspace(0, 1.1, len(labels4))]
    # 将排列在第4位的语言(Python)分离出来
    explode4 = [0, 0, 0, 0, 0, 0.02, 0.04, 0.06, 0.1, 0.12]
    # 旋转角度
    plt.title("The distance from the relationship word to E2", fontsize=20)
    # 标题
    plt.pie(values4, labels=labels4, explode=explode4, colors=colors4, pctdistance=0.9,
            startangle=180,
            shadow=True, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()
    print(labels4)

    ##########关系词的长度#############
    Relationwords_length = []
    noun_length_all_1 = []
    verb_length_all_1 = []
    prep_length_all_1 = []
    other_length_all_1 = []
    noun_length_all_2 = []
    verb_length_all_2 = []
    prep_length_all_2 = []
    other_length_all_2 = []
    
    for i in batches:
        noun_length = []
        verb_length = []
        prep_length = []
        other_length = []
        for j in i:
            if j[5] == 'unknown':
                continue
            if j[6] in ['f','ns','n','nr','nt']:
                length = len(j[4])
                noun_length.append(length)
            elif j[6] == 'v':
                length = len(j[4])
                verb_length.append(length)
            elif j[6] == 'p':
                length = len(j[4])
                prep_length.append(length)
            else:
                length = len(j[4])
                other_length.append(length)
    
        noun_length_min = min(noun_length)
        noun_length_max = max(noun_length)
        noun_length_all_1.append(noun_length_min)
        noun_length_all_2.append(noun_length_max)
    
        verb_length_min = min(verb_length)
        verb_length_max = max(verb_length)
        verb_length_all_1.append(verb_length_min)
        verb_length_all_2.append(verb_length_max)
    
        if len(prep_length) > 0:
            prep_length_min = min(prep_length)
            prep_length_max = max(prep_length)
            prep_length_all_1.append(prep_length_min)
            prep_length_all_2.append(prep_length_max)
    
        if len(other_length) > 0:
           other_length_min = min(other_length)
           other_length_max = max(other_length)
           other_length_all_1.append(other_length_min)
           other_length_all_2.append(other_length_max)
    
    noun_min_length_average = min(noun_length_all_1)
    noun_max_length_average = max(noun_length_all_2)
    verb_min_length_average = min(verb_length_all_1)
    verb_max_length_average = max(verb_length_all_2)
    prep_min_length_average = min(prep_length_all_1)
    prep_max_length_average = max(prep_length_all_2)
    other_min_length_average = min(other_length_all_1)
    other_max_length_average = max(other_length_all_2)

    ##########e1到e2的距离##############
    E2_dis_e1 = []
    for i in batches:
        e2_dis_e1 = []
        for j in i:
            e2_e1 = (int(j[3]) - int(j[2]))
            e2_dis_e1.append(e2_e1)
        E2_dis_e1.append(e2_dis_e1)
    
    dis_min = []
    dis_max = []
    for i in E2_dis_e1:
        dis_min_1 = min(i)
        dis_max_1 = max(i)
        dis_min.append(dis_min_1)
        dis_max.append(dis_max_1)
    
    dis_min_average = np.mean(dis_min)
    dis_max_average = np.mean(dis_max)


    return yuanwen


if __name__ == '__main__':
    inputs = open('Data/train_data_pre_100_label.txt', 'r', encoding='utf-8')
    Bootstrapping(inputs)
