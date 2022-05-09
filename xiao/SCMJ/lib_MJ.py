# ！/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     : 2021/10/21 15:41
# @Author   : Zou
# @Email    : 1091274580@qq.com
# @File     : lib_MJ.py
# @Software : PyCharm

'''
本文件为四川麻将函数相关库
'''

import copy
# import numpy as np
import time
import random

# 牌值为0x00 ~ 0x29共27张
cards_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25, 33, 34, 35, 36, 37, 38, 39, 40, 41]

t2s = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [17, 17], [18, 18], [19, 19], [20, 20],
       [21, 21], [22, 22], [23, 23], [24, 24], [25, 25], [33, 33], [34, 34], [35, 35], [36, 36], [37, 37], [38, 38],
       [39, 39], [40, 40], [41, 41], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [17, 18], [18, 19],
       [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38],
       [38, 39], [39, 40], [40, 41], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9], [17, 19], [18, 20], [19, 21],
       [20, 22], [21, 23], [22, 24], [23, 25], [33, 35], [34, 36], [35, 37], [36, 38], [37, 39], [38, 40], [39, 41],
       [1, 1, 2], [1, 1, 3], [1, 2, 2], [2, 2, 3], [2, 2, 4], [1, 3, 3], [2, 3, 3], [3, 3, 4], [3, 3, 5], [2, 4, 4],
       [3, 4, 4], [4, 4, 5],
       [4, 4, 6], [3, 5, 5], [4, 5, 5], [5, 5, 6], [5, 5, 7], [4, 6, 6], [5, 6, 6], [6, 6, 7], [6, 6, 8], [5, 7, 7],
       [6, 7, 7], [7, 7, 8], [7, 7, 9], [6, 8, 8], [7, 8, 8], [8, 8, 9], [7, 9, 9], [8, 9, 9], [17, 17, 18],
       [17, 17, 19], [17, 18, 18], [18, 18, 19], [18, 18, 20], [17, 19, 19], [18, 19, 19], [19, 19, 20], [19, 19, 21],
       [18, 20, 20], [19, 20, 20], [20, 20, 21], [20, 20, 22], [19, 21, 21], [20, 21, 21], [21, 21, 22], [21, 21, 23],
       [20, 22, 22], [21, 22, 22], [22, 22, 23], [22, 22, 24], [21, 23, 23], [22, 23, 23], [23, 23, 24], [23, 23, 25],
       [22, 24, 24], [23, 24, 24], [24, 24, 25], [23, 25, 25], [24, 25, 25], [33, 33, 34], [33, 33, 35], [33, 34, 34],
       [34, 34, 35], [34, 34, 36], [33, 35, 35], [34, 35, 35], [35, 35, 36], [35, 35, 37], [34, 36, 36], [35, 36, 36],
       [36, 36, 37], [36, 36, 38], [35, 37, 37], [36, 37, 37], [37, 37, 38], [37, 37, 39], [36, 38, 38], [37, 38, 38],
       [38, 38, 39], [38, 38, 40], [37, 39, 39], [38, 39, 39], [39, 39, 40], [39, 39, 41], [38, 40, 40], [39, 40, 40],
       [40, 40, 41], [39, 41, 41], [40, 41, 41], [1, 2, 4], [2, 3, 5], [1, 3, 4], [3, 4, 6], [2, 4, 5], [4, 5, 7],
       [3, 5, 6], [5, 6, 8], [4, 6, 7], [6, 7, 9], [5, 7, 8], [6, 8, 9], [17, 18, 20], [18, 19, 21], [17, 19, 20],
       [19, 20, 22], [18, 20, 21], [20, 21, 23], [19, 21, 22], [21, 22, 24], [20, 22, 23], [22, 23, 25], [21, 23, 24],
       [22, 24, 25], [33, 34, 36], [34, 35, 37], [33, 35, 36], [35, 36, 38], [34, 36, 37], [36, 37, 39], [35, 37, 38],
       [37, 38, 40], [36, 38, 39], [38, 39, 41], [37, 39, 40], [38, 40, 41], [1, 3, 5], [2, 4, 6], [3, 5, 7], [4, 6, 8],
       [5, 7, 9], [17, 19, 21], [18, 20, 22], [19, 21, 23], [20, 22, 24], [21, 23, 25], [33, 35, 37], [34, 36, 38],
       [35, 37, 39], [36, 38, 40], [37, 39, 41]]

'''
四川麻将中因为不能吃，只能碰，所以aa和ab的权重一样
'''
w_aa = 2
w_ab = 2

def splitColor(cards):
    color = [[], [], []]
    for card in cards:
        if card & 0xf0 == 0:
            color[0].append(card)
        elif card & 0xf0 == 0x10:
            color[1].append(card)
        elif card & 0xf0 == 0x20:
            color[2].append(card)
    return color


def get_index(list=[], n=[]):
    list = copy.copy(list)
    n = copy.copy(n)
    index = []
    j_used = []
    for i in n:
        for j in range(len(list)):
            if i == list[j] and j not in j_used:
                index.append(j)
                j_used.append(j)
    return index


def split_type_s(cards=[]):
    """
    功能：手牌花色分离，将手牌分离成万条筒字各色后输出
    :param cards: 手牌　[]
    :return: 万,条,筒,字　[],[],[],[]
    """
    cards_wan = []
    cards_tiao = []
    cards_tong = []
    for card in cards:
        if card & 0xF0 == 0x00:
            cards_wan.append(card)
        elif card & 0xF0 == 0x10:
            cards_tiao.append(card)
        elif card & 0xF0 == 0x20:
            cards_tong.append(card)
    return cards_wan, cards_tiao, cards_tong


def get_effective_cards(dz):
    """
    功能：计算搭子的有效牌
    :param dz:
    :return:
    """
    effect_cards = []
    if dz[0] == dz[1]:
        effect_cards.append(dz[0])
    elif dz[0] + 2 == dz[1]:
        effect_cards.append(dz[0]+1)
    else:
        if dz[0] - 1 & 0x0F > 0:
            effect_cards.append(dz[0]-1)
        if dz[1] + 1 & 0x0F <= 9:
            effect_cards.append(dz[1]+1)
    return effect_cards


def get_32N(cards=[]):
    """
    功能：计算所有存在的手牌的３Ｎ与２Ｎ的集合，例如[3,4,5]　，将得到[[3,4],[3,5],[4,5],[3,4,5]]
    思路：为减少计算量，对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
    :param cards: 手牌　[]
    :return: 3N与2N的集合　[[]]
    """
    cards.sort()
    kz = []
    sz = []
    aa = []
    ab = []
    ac = []
    lastCard = 0
    # 对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
    if len(cards) >= 12:
        for card in cards:
            if card == lastCard:
                continue
            else:
                lastCard = card
            if cards.count(card) >= 3:
                kz.append([card, card, card])
            elif cards.count(card) >= 2:
                aa.append([card, card])
            if card + 1 in cards and card + 2 in cards:
                sz.append([card, card + 1, card + 2])
            else:
                if card + 1 in cards:
                    ab.append([card, card + 1])
                if card + 2 in cards:
                    ac.append([card, card + 2])
    else:
        for card in cards:
            if card == lastCard:
                continue
            else:
                lastCard = card
            if cards.count(card) >= 3:
                kz.append([card, card, card])
            if cards.count(card) >= 2:
                aa.append([card, card])
            if card + 1 in cards and card + 2 in cards:
                sz.append([card, card + 1, card + 2])
            if card + 1 in cards:
                ab.append([card, card + 1])
            if card + 2 in cards:
                ac.append([card, card + 2])
    return kz + sz + aa + ab + ac

    # 判断３２Ｎ是否存在于ｃａｒｄｓ中


def in_cards(t32=[], cards=[]):
    """
    判断３２Ｎ是否存在于cards中
    :param t32: 3N或2N组合牌
    :param cards: 本次判断的手牌
    :return: bool
    """
    for card in t32:
        if card not in cards:
            return False
    return True


def extract_32N(cards=[], t32_branch=[], t32_set=[]):
    """
    功能：递归计算手牌的所有组合信息，并存储在t32_set，
    思路: 每次递归前检测是否仍然存在３２N的集合,如果没有则返回出本此计算的结果，否则在手牌中抽取该３２N，再次进行递归
    :param cards: 手牌
    :param t32_branch: 本次递归的暂存结果
    :param t32_set: 所有组合信息
    :return: 结果存在t32_set中
    """
    t32N = get_32N(cards=cards)

    if len(t32N) == 0:
        t32_set.extend(t32_branch)
        # t32_set.extend([cards])
        t32_set.append(0)
        t32_set.extend([cards])
    else:
        for t32 in t32N:
            if in_cards(t32=t32, cards=cards):
                cards_r = copy.copy(cards)
                for card in t32:
                    cards_r.remove(card)
                t32_branch.append(t32)
                extract_32N(cards=cards_r, t32_branch=t32_branch, t32_set=t32_set)
                if len(t32_branch) >= 1:
                    t32_branch.pop(-1)


def tree_expand(cards):
    """
    功能：对extract_32N计算的结果进行处理同一格式，计算万条筒花色的组合信息
    思路：对t32_set的组合信息进行格式统一，分为[kz,sz,aa,ab,xts,leftCards]保存，并对划分不合理的地方进行过滤，例如将３４５划分为35,4为废牌的情况

    :param cards: cards [] 万条筒其中一种花色手牌
    :return: allDeWeight　[kz,sz,aa,ab,xts,leftCards] 去除不合理划分情况的组合后的组合信息
    """
    all = []
    t32_set = []
    extract_32N(cards=cards, t32_branch=[], t32_set=t32_set)
    kz = []
    sz = []
    t2N = []
    aa = []
    length_t32_set = len(t32_set)
    i = 0
    while i < length_t32_set:
        t = t32_set[i]
        flag = True  # 本次划分是否合理
        if t != 0:
            if len(t) == 3:

                if t[0] == t[1]:
                    kz.append(t)
                else:
                    sz.append(t)  # print (sub)
            elif len(t) == 2:
                if t[1] == t[0]:
                    aa.append(t)
                else:
                    t2N.append(t)

        else:
            '修改，使计算时间缩短'
            leftCards = t32_set[i + 1]
            efc_cards = get_effective_cards(dz_set=t2N)  # t2N中不包含ａａ
            # 去除划分不合理的情况，例如345　划分为34　或35等，对于333 划分为33　和3的情况，考虑有将牌的情况暂时不做处理
            for card in leftCards:
                if card in efc_cards:
                    flag = False
                    break

            if flag:
                all.append([kz, sz, aa, t2N, 0, leftCards])
            kz = []
            sz = []
            aa = []
            t2N = []
            i += 1
        i += 1

    allSort = []  # 给每一个元素排序
    allDeWeight = []  # 排序去重后

    for e in all:
        for f in e:
            if f == 0:  # 0是xts位，int不能排序
                continue
            else:
                f.sort()
        allSort.append(e)

    for a in allSort:
        if a not in allDeWeight:
            allDeWeight.append(a)

    allDeWeight = sorted(allDeWeight, key=lambda k: (len(k[0]), len(k[1]), len(k[2])), reverse=True)  # 居然可以这样排序！！
    return allDeWeight


def assess_card(card):
    value = 0
    if card & 0x0f == 1 or card & 0x0f == 9:
        value = 3
    elif card & 0x0f == 2 or card & 0x0f == 8:
        value = 4
    elif card & 0x0f == 3 or card & 0x0f == 7:
        value = 8
    elif card & 0x0f == 4 or card & 0x0f == 6:
        value = 5
    elif card & 0x0f == 5:
        value = 6
    return value


def translate16_33(i):
    i = int(i)
    if i >= 0x01 and i <= 0x09:
        i = i - 1
    elif i >= 0x11 and i <= 0x19:
        i = i - 8
    elif i >= 0x21 and i <= 0x29:
        i = i - 15
    elif i >= 0x31 and i <= 0x37:
        i = i - 22
    else:
        print ("translate16_33 is error,i=%d" % i)
        i = -1
    return i


def translate33_16(i):  # 将下标转换成16进制的牌
    if 0 <= i < 9:
        return i + 1
    elif 9 <= i < 18:
        return i + 8
    elif 18 <= i < 27:
        return i + 15
    elif 27 <= i < 34:
        return i + 22
    else:
        print("[INFO_ZW]:INPUT ERROR")


def translate33_10(i):  # 转换成10进制
    if 0 <= i < 9:
        return i + 1
    elif 9 <= i < 18:
        return i + 2
    elif 18 <= i < 27:
        return i + 3
    elif 27 <= i < 34:
        return i + 4
    else:
        print("[INFO_ZW]:INPUT ERROR")


def translate16_10(i):
    # 16进制转进制
    return i // 16 * 10 + i % 16


# 把对应的十六进制牌转换成[]*34的索引值
# 如0x01代表的是第0个数
def convert_hex2index(a):
    if a > 0 and a < 0x10:
        return a - 1
    if a > 0x10 and a < 0x20:
        return a - 8
    if a > 0x20 and a < 0x30:
        return a - 15
    if a > 0x30 and a < 0x40:
        return a - 22

# 获取ｌｉｓｔ中的最小值和下标
def get_min(list=[]):
    min = 14
    index = 0
    for i in range(len(list)):
        if list[i] < min:
            min = list[i]
            index = i
    return min, index

def get_t2info():
    """"
    功能：计算所有的搭子，并计算对应的搭子的有效牌，同时计算出有效牌的指示下标
    """
    dzSet = [0] * (34 + 15 * 3)  # 34+15*3
    # 生成搭子有效牌表
    dzEfc = [0] * (34 + 15 * 3)
    for i in range(len(dzSet)):
        if i <= 33:  # aa
            card = int(i / 9) * 16 + i % 9 + 1
            dzSet[i] = [card, card]
            dzEfc[i] = [card]
        elif i <= 33 + 8 * 3:  # ab
            card = int((i - 34) / 8) * 16 + (i - 34) % 8 + 1
            dzSet[i] = [card, card + 1]
            if card & 0x0f == 1:
                dzEfc[i] = [card + 2]
            elif card & 0x0f == 8:
                dzEfc[i] = [card - 1]
            else:
                dzEfc[i] = [card - 1, card + 2]
        else:
            card = int((i - 34 - 8 * 3) / 7) * 16 + (i - 34 - 8 * 3) % 7 + 1
            dzSet[i] = [card, card + 2]
            dzEfc[i] = [card + 1]

    efc_dzindex = {}  # card->34+8+8+8+7+7+7
    cardSet = []
    for i in range(34):
        cardSet.append(i // 9 * 16 + i % 9 + 1)
    for card in cardSet:
        efc_dzindex[card] = []
        efc_dzindex[card].append(translate16_33(card))  # 加aa
        color = int(card / 16)
        if color != 3:
            if card & 0x0f == 1:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) + 1)

            elif card & 0x0f == 2:  # 13 34
                efc_dzindex[card].append(33 + 24 + color * 7 + (card & 0x0f) - 1)
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) + 1)
            elif card & 0x0f == 8:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) - 2)
                efc_dzindex[card].append(33 + 24 + color * 7 + (card & 0x0f) - 1)
            elif card & 0x0f == 9:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) - 2)
            else:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) - 2)
                efc_dzindex[card].append(33 + 24 + color * 7 + (card & 0x0f) - 1)
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) + 1)
    return dzSet, dzEfc, efc_dzindex

def get_t3info():
    """
    功能：计算所有的刻子和顺子
    """
    t3Set = []
    for i in range(27):
        card = int(i / 9) * 16 + i % 9 + 1
        t3Set.append([card, card, card])
    for i in range(27, 27 + 7 * 3):
        card = int((i - 27) / 7) * 16 + (i - 27) % 7 + 1
        t3Set.append([card, card + 1, card + 2])
    return t3Set


dzSet, dzEfc, efc_dzindex = get_t2info()
t3Set = get_t3info()

def t2tot3_info():
    """
    生成t2转化为t3的状态
    :return: "t2": [[t2_,t3,t1_left,valid,p]]
    """
    t2tot3_dict = {}
    for t2 in t2s:
        t2tot3_dict[str(t2)] = []
        if len(t2) == 2:
            t2_decompose_valid = [t2]
            t1_left = [[]]

        elif len(t2) == 3:
            t2_decompose = [[t2[0], t2[1]], [t2[0], t2[2]], [t2[1], t2[2]]]  # 存储拆分后的t2
            t1 = [[t2[2]], [t2[1]], [t2[0]]]
            t2_decompose_valid = []
            t1_left = []
            for i in range(len(t2_decompose)):
                t2_ = t2_decompose[i]
                if abs(t2_[1] - t2_[0]) <= 2:
                    if t2_ not in t2_decompose_valid:
                        t2_decompose_valid.append(t2_)
                        t1_left.append(t1[i])

        for j in range(len(t2_decompose_valid)):
            t2_ = t2_decompose_valid[j]
            valids = dzEfc[dzSet.index(t2_)]
            for valid_card in valids:
                info = []
                t3 = copy.copy(t2_)
                t3.append(valid_card)
                t3.sort()
                info.append(t2_)
                info.append(t3)
                info.append(t1_left[j])
                info.append(valid_card)

                index = convert_hex2index(valid_card)
                if t2_[0] == t2_[1]:
                    info.append(w_aa)
                else:
                    info.append(w_ab)
                t2tot3_dict[str(t2)].append(info)
    return t2tot3_dict

def t1tot2_info():
    """
    t1转换为t2的状态
    :return:
    """
    t1tot2_dict = {}
    for card in cards_value:
        t1tot2_dict[str(card)] = []
        if card < 0x31:
            valid_tile = [card - 2, card - 1, card, card + 1, card + 2]
        else:
            valid_tile = [card]
        # t2_transform=[]
        for tile in valid_tile:
            if tile in cards_value:
                t2 = sorted([card, tile])
                info = []
                info.append(t2)
                info.append(tile)
                info.append(1)
                t1tot2_dict[str(card)].append(info)
    return t1tot2_dict

def t1tot3_info():
    """

    :param T_selfmo:
    :param RT1:
    :param RT2:
    :param RT3:
    :return: {"t1":[[t3,t2(valid card),p]]}

    """
    t1tot3_dict = {}
    for card in cards_value:
        t1tot3_dict[str(card)] = []
        if card < 0x31:
            valid_tiles = [[card - 2, card - 1], [card - 1, card + 1], [card, card], [card + 1, card + 2]]
        else:
            valid_tiles = [[card, card]]
        for t2 in valid_tiles:
            if t2[0] in cards_value and t2[1] in cards_value:
                info = []
                t3 = copy.copy(t2)
                t3.append(card)
                t3.sort()
                info.append(t3)
                info.append(t2)

                if t2[0] == t2[1]:
                    info.append([1, w_aa])
                else:
                    info.append([1, w_ab])
                t1tot3_dict[str(card)].append(info)
    return t1tot3_dict

def is_1l_list(l):
    for i in l:
        if type(i) == list:
            return False
    return True

def deepcopy(src):
    dst = []
    for i in src:
        if type(i) == list and not is_1l_list(i):
            i = deepcopy(i)
        dst.append(copy.copy(i))
    return dst


def cal_xts(all=[], suits=[], isSpecial=False):
    """
     功能：计算组合的向听数
    思路：初始向听数为14，减去相应已成型的组合（kz,sz为３，aa/ab为２），当２Ｎ过剩时，只减去还需要的２Ｎ，对２Ｎ不足时，对还缺少的３Ｎ减去１，表示从孤张牌中选择一张作为３Ｎ的待选
    :param all: [[]]组合信息
    :param suits: 副露
    :return: all　计算向听数后的组合信息
    """
    for i in range(len(all)):
        t3N = all[i][0] + all[i][1]
        all[i][4] = 14 - (len(t3N) + len(suits)) * 3
        # 有将牌
        has_aa = False
        if len(all[i][2]) > 0:
            has_aa = True
        if has_aa:  # has do 当２Ｎ与３Ｎ数量小于4时，存在没有减去相应待填数，即废牌也会有１张作为２Ｎ或３Ｎ的待选位,
            if len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) - 1 >= 4:
                all[i][4] -= (4 - (len(suits) + len(t3N))) * 2 + 2
            else:
                all[i][4] -= (len(all[i][2]) + len(all[i][3]) - 1) * 2 + 2 + 4 - (
                        len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) - 1)
        # 无将牌
        else:
            if len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) >= 4:
                all[i][4] -= (4 - (len(suits) + len(t3N))) * 2 + 1
            else:
                all[i][4] -= (len(all[i][2]) + len(all[i][3])) * 2 + 1 + 4 - (
                        len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]))
        if isSpecial:

            special_cards = len(all[i][-1])
            if special_cards > all[i][4]:
                all[i][4] = special_cards
        if all[i][4] < 0:
            all[i][4] = 0
    all.sort(key=lambda k: (k[4], len(k[-2])))
    return all

'''
平胡类型相关处理方法，包括屁胡、清一色、幺九、断幺九
分为手牌拆分模块sys_info，评估cost,出牌决策，吃碰杠决策等部分
'''
class PingHu:
    def __init__(self, cards, suits, padding=[], fan_type=0, useless_cards=[]):
        """
        类变量初始化
        :param cards: 手牌　
        :param suits:副露
        :param padding: 扩展牌，动作决策时用
        :param fan_type: 番型种类： {0：屁胡， 1：清一色， 2：幺九， 3：断幺九}
        :return  平胡：[[kz], [sz], [aa], [t2], xts, [t1], [useless]]
                 清一色：[[kz], [sz], [aa], [t2], xts, [t1], [useless]]
                 幺九： [[kz], [sz], [aa], [t2], xts, [t1], [useless]]
                 断幺九： [[kz], [sz], [aa], [t2], xts, [t1], [useless]]
        """
        cards.sort()
        self.cards = cards
        self.suits = suits
        self.padding = padding
        self.fan_type = fan_type
        self.useless_cards = useless_cards

    @staticmethod
    def split_type_s(cards=[]):
        """
        功能：手牌花色分离，将手牌分离成万条筒字各色后输出
        :param cards: 手牌　[]
        :return: 万,条,筒,字　[],[],[],[]
        """
        cards_wan = []
        cards_tiao = []
        cards_tong = []
        for card in cards:
            if card & 0xF0 == 0x00:
                cards_wan.append(card)
            elif card & 0xF0 == 0x10:
                cards_tiao.append(card)
            elif card & 0xF0 == 0x20:
                cards_tong.append(card)
        return cards_wan, cards_tiao, cards_tong

    @staticmethod
    def get_effective_cards(dz_set=[]):
        """
        获取有效牌
        :param dz_set: 搭子集合 list [[]]
        :return: 有效牌 list []
        """
        effective_cards = []
        for dz in dz_set:
            if len(dz) == 1:
                effective_cards.append(dz[0])
            elif dz[1] == dz[0]:
                effective_cards.append(dz[0])
            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:
                    effective_cards.append(dz[0] + 2)
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append((dz[0] - 1))
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
            elif dz[1] == dz[0] + 2:
                effective_cards.append(dz[0] + 1)
        effective_cards = set(effective_cards)  # set 和list的区别？
        return list(effective_cards)

    # 判断３２Ｎ是否存在于ｃａｒｄｓ中
    @staticmethod
    def in_cards(t32=[], cards=[]):
        """
        判断３２Ｎ是否存在于ｃａｒｄｓ中
        :param t32: ３Ｎ或2N组合牌
        :param cards: 本次判断的手牌
        :return: bool
        """
        for card in t32:
            if card not in cards:
                return False
        return True

    @staticmethod
    def get_32N(cards=[]):
        """
        功能：计算所有存在的手牌的３Ｎ与２Ｎ的集合，例如[3,4,5]　，将得到[[3,4],[3,5],[4,5],[3,4,5]]
        思路：为减少计算量，对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
        :param cards: 手牌　[]
        :return: 3N与2N的集合　[[]]
        """
        cards.sort()
        kz = []
        sz = []
        aa = []
        ab = []
        ac = []
        lastCard = 0
        # 对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
        if len(cards) >= 8:
            for card in cards:
                if card == lastCard:
                    continue
                else:
                    lastCard = card
                if cards.count(card) >= 3:
                    kz.append([card, card, card])
                elif cards.count(card) >= 2:
                    aa.append([card, card])
                if card + 1 in cards and card + 2 in cards:
                    sz.append([card, card + 1, card + 2])
                else:
                    if card + 1 in cards:
                        ab.append([card, card + 1])
                    if card + 2 in cards:
                        ac.append([card, card + 2])
        else:
            for card in cards:
                if card == lastCard:
                    continue
                else:
                    lastCard = card
                if cards.count(card) >= 3:
                    kz.append([card, card, card])
                if cards.count(card) >= 2:
                    aa.append([card, card])
                if card + 1 in cards and card + 2 in cards:
                    sz.append([card, card + 1, card + 2])
                if card + 1 in cards:
                    ab.append([card, card + 1])
                if card + 2 in cards:
                    ac.append([card, card + 2])
        return kz + sz + aa + ab + ac

    def extract_32N(self, cards=[], t32_branch=[], t32_set=[]):
        """
        功能：递归计算手牌的所有组合信息，并存储在t32_set，
        思路: 每次递归前检测是否仍然存在３２N的集合,如果没有则返回出本此计算的结果，否则在手牌中抽取该３２N，再次进行递归
        :param cards: 手牌
        :param t32_branch: 本次递归的暂存结果
        :param t32_set: 所有组合信息
        :return: 结果存在t32_set中
        """
        t32N = self.get_32N(cards=cards)
        if len(t32N) == 0:
            t32_set.extend(t32_branch)
            t32_set.append(0)
            t32_set.extend([cards])
        else:
            for t32 in t32N:
                if self.in_cards(t32=t32, cards=cards):
                    cards_r = copy.copy(cards)
                    for card in t32:
                        cards_r.remove(card)
                    t32_branch.append(t32)
                    self.extract_32N(cards=cards_r, t32_branch=t32_branch, t32_set=t32_set)
                    if len(t32_branch) >= 1:
                        t32_branch.pop(-1)

    def tree_expand(self, cards):
        """
        功能：对extract_32N计算的结果进行处理同一格式，计算万条筒花色的组合信息
        思路：对t32_set的组合信息进行格式统一，分为[kz,sz,aa,ab,xts,leftCards]保存，并对划分不合理的地方进行过滤，例如将３４５划分为35,4为废牌的情况
        :param cards: cards [] 万条筒其中一种花色手牌
        :return: allDeWeight　[kz,sz,aa,ab,xts,leftCards] 去除不合理划分情况的组合后的组合信息
        """
        all = []
        t32_set = []
        self.extract_32N(cards=cards, t32_branch=[], t32_set=t32_set)
        kz = []
        sz = []
        t2N = []
        aa = []
        length_t32_set = len(t32_set)
        i = 0
        while i < length_t32_set:
            t = t32_set[i]
            flag = True  # 本次划分是否合理
            if t != 0:
                if len(t) == 3:
                    if t[0] == t[1]:
                        kz.append(t)
                    else:
                        sz.append(t)
                elif len(t) == 2:
                    if t[1] == t[0]:
                        aa.append(t)
                    else:
                        t2N.append(t)
            else:
                '修改，使计算时间缩短'
                leftCards = t32_set[i + 1]
                efc_cards = self.get_effective_cards(dz_set=t2N)  # t2N中不包含ａａ
                # 去除划分不合理的情况，例如345　划分为34　或35等，对于333 划分为33　和3的情况，考虑有将牌的情况暂时不做处理
                for card in leftCards:
                    if card in efc_cards:
                        flag = False
                        break
                if flag:
                    all.append([kz, sz, aa, t2N, 0, leftCards])
                kz = []
                sz = []
                aa = []
                t2N = []
                i += 1
            i += 1
        allSort = []  # 给每一个元素排序
        allDeWeight = []  # 排序去重后

        for e in all:
            for f in e:
                if f == 0:  # 0是xts位，int不能排序
                    continue
                else:
                    f.sort()
            allSort.append(e)

        for a in allSort:
            if a not in allDeWeight:
                allDeWeight.append(a)
        allDeWeight = sorted(allDeWeight, key=lambda k: (len(k[0]), len(k[1]), len(k[2])), reverse=True)
        return allDeWeight

    def pinghu_CS(self, cards=[], suits=[], t1=[]):
        """
        功能：综合计算手牌的组合信息
        思路：对手牌进行花色分离后，单独计算出每种花色的组合信息，再将其综合起来，计算每个组合向听数，最后输出最小向听数及其加一的组合
        :param cards: 手牌
        :param suits: 副露
        :param t1: 剩余牌
        :return: 组合信息 [[kz], [sz], [aa], [ab/ac], xts, [t1], [useless_cards]]
        """
        if cards==[]:
            cards = copy.copy(self.cards)
            suits = deepcopy(self.suits)

        yaojiu_cards = [0x01, 0x09, 0x11, 0x19, 0x21, 0x29]
        if self.fan_type == 2:  # 判断副露是否符合幺九牌型
            for suit in suits:
                if suit[0] == suit[1] and suit[0] not in yaojiu_cards:
                    return [[[], [], [], [], 14, [], []]]
                if suit[0] != suit[1] and suit[0] not in yaojiu_cards and suit[2] not in yaojiu_cards:
                    return [[[], [], [], [], 14, [], []]]
            # 副露符合，将手牌中的废牌加入到useless集合中
            for card in cards:
                if card & 0x0F in [4, 5, 6]:
                    self.useless_cards.append(card)
            for card in self.useless_cards:
                cards.remove(card)

        if self.fan_type == 3:  # 判断副露是否符合断幺九牌型
            for suit in suits:
                for card in suit:
                    if card in yaojiu_cards:
                        return [[[], [], [], [], 14, [], []]]
            # 副露符合，将手牌中的幺九牌加入到useless集合中
            for card in cards:
                if card & 0x0F in [1, 9]:
                    self.useless_cards.append(card)
            for card in self.useless_cards:
                cards.remove(card)

        # 花色分离
        wan, tiao, tong = self.split_type_s(cards=cards)
        wan_expd = self.tree_expand(cards=wan)
        tiao_expd = self.tree_expand(cards=tiao)
        tong_expd = self.tree_expand(cards=tong)

        all = []
        for i in wan_expd:
            for j in tiao_expd:
                for k in tong_expd:
                        branch = []
                        # 将每种花色的4个字段合并成一个字段
                        for n in range(6):
                            branch.append(i[n] + j[n] + k[n])
                        branch[-1] += self.padding+t1
                        all.append(branch)

        if self.fan_type == 2:   # yaojiu
            for a in all:
                invalid_kz = []
                for kz in a[0]:  # 非幺九刻子放到废牌中
                    if kz[0] not in yaojiu_cards:
                        invalid_kz.append(kz)
                for kz in invalid_kz:
                    a[0].remove(kz)
                    a[-1] += kz
                invalid_aa = []
                for aa in a[2]:  # 非幺九对子放到废牌中
                    if aa[0] not in yaojiu_cards:
                        invalid_aa.append(aa)
                for aa in invalid_aa:
                    a[2].remove(aa)
                    a[-1] += aa
        for a in all:
            a.append(self.useless_cards)

        # 计算向听数
        # 计算拆分组合的向听数
        isSpecial = False   # 清一色、断幺九、幺九牌的向听数计算方式不一样
        if self.fan_type == 1 or self.fan_type == 2 or self.fan_type == 3:
            isSpecial = True
        all = cal_xts(all, suits, isSpecial)


        # 获取向听数最小的all分支
        min_index = 0
        for i in range(len(all)):
            if all[i][4] > all[0][4]:  # xts+1以下的组合
                min_index = i
                break

        if min_index == 0:  # 如果全部都匹配，则min_index没有被赋值，将min_index赋予all长度
            min_index = len(all)

        all = all[:min_index]

        #处理向听数为0时的情况，需要从中依次选择一张牌作为t1
        if all[0][-2] == 0 and all[0][-1] == []:
            all = []
            for card in list(set(cards)):
                cards_ = copy.copy(cards)
                cards_.remove(card)
                all += self.pinghu_CS(cards=cards_, suits=suits, t1=[card])
        return all


def get_effective_cards_gang(dz_set=[]):
    """
     switch使用，
    获取有效牌
    :param dz_set: 搭子集合 list [[]]
    :return: 有效牌 list []
    """
    effective_cards = []
    for dz in dz_set:
        if len(dz) == 1:
            effective_cards.append(dz[0])
        elif dz[1] == dz[0]:
            effective_cards.append(dz[0])
        elif dz[1] == dz[0] + 1:
            if int(dz[0]) & 0x0F == 1:
                effective_cards.append(dz[0] + 2)
            elif int(dz[0]) & 0x0F == 8:
                effective_cards.append((dz[0] - 1))
            else:
                effective_cards.append(dz[0] - 1)
                effective_cards.append(dz[0] + 2)
        elif dz[1] == dz[0] + 2:
            effective_cards.append(dz[0] + 1)
    effective_cards = set(effective_cards)  # set 和list的区别？
    return list(effective_cards)


def get_32N_gang(cards=[]):
    """
    功能：计算所有存在的手牌的３Ｎ与２Ｎ的集合，例如[3,4,5]　，将得到[[3,4],[3,5],[4,5],[3,4,5]]
    思路：为减少计算量，对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
    :param cards: 手牌　[]
    :return: 3N与2N的集合　[[]]
    """
    cards.sort()
    gang = []
    kz = []
    sz = []
    aa = []
    ab = []
    ac = []
    lastCard = 0
    # 对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
    if len(cards) >= 12:
        for card in cards:
            if card == lastCard:
                continue
            else:
                lastCard = card
            if cards.count((card)) == 4:
                gang.append([card, card, card, card])
            if cards.count(card) >= 3:
                kz.append([card, card, card])
            elif cards.count(card) >= 2:
                aa.append([card, card])
            if card + 1 in cards and card + 2 in cards:
                sz.append([card, card + 1, card + 2])
            else:
                if card + 1 in cards:
                    ab.append([card, card + 1])
                if card + 2 in cards:
                    ac.append([card, card + 2])
    else:
        for card in cards:
            if card == lastCard:
                continue
            else:
                lastCard = card
            if cards.count(card) == 4:
                gang.append([card, card, card, card])
            if cards.count(card) >= 3:
                kz.append([card, card, card])
            if cards.count(card) >= 2:
                aa.append([card, card])
            if card + 1 in cards and card + 2 in cards:
                sz.append([card, card + 1, card + 2])
            if card + 1 in cards:
                ab.append([card, card + 1])
            if card + 2 in cards:
                ac.append([card, card + 2])
    return gang + kz + sz + aa + ab + ac


def in_cards_gang(t32=[], cards=[]):
    """
     switch使用，
    判断３２Ｎ是否存在于ｃａｒｄｓ中
    :param t32: ３Ｎ或2N组合牌
    :param cards: 本次判断的手牌
    :return: bool
    """
    for card in t32:
        if card not in cards:
            return False
    return True


def extract_32N_gang(cards=[], t32_branch=[], t32_set=[]):
    """
    功能：递归计算手牌的所有组合信息，并存储在t32_set，
    思路: 每次递归前检测是否仍然存在３２N的集合,如果没有则返回出本此计算的结果，否则在手牌中抽取该３２N，再次进行递归
    :param cards: 手牌
    :param t32_branch: 本次递归的暂存结果
    :param t32_set: 所有组合信息
    :return: 结果存在t32_set中
    """
    t32N = get_32N_gang(cards=cards)

    if len(t32N) == 0:
        t32_set.extend(t32_branch)
        # t32_set.extend([cards])
        t32_set.append(0)
        t32_set.extend([cards])
    else:
        for t32 in t32N:
            if in_cards_gang(t32=t32, cards=cards):
                cards_r = copy.copy(cards)
                for card in t32:
                    cards_r.remove(card)
                t32_branch.append(t32)
                extract_32N_gang(cards=cards_r, t32_branch=t32_branch, t32_set=t32_set)
                if len(t32_branch) >= 1:
                    t32_branch.pop(-1)


def tree_expand_gang(cards):
    """
    功能：对extract_32N计算的结果进行处理同一格式，计算万条筒花色的组合信息
    思路：对t32_set的组合信息进行格式统一，分为[kz,sz,aa,ab,xts,leftCards]保存，并对划分不合理的地方进行过滤，例如将３４５划分为35,4为废牌的情况

    :param cards: cards [] 万条筒其中一种花色手牌
    :return: allDeWeight　[kz,sz,aa,ab,xts,leftCards] 去除不合理划分情况的组合后的组合信息
    """
    all = []
    t32_set = []
    extract_32N_gang(cards=cards, t32_branch=[], t32_set=t32_set)
    gang = []
    kz = []
    sz = []
    t2N = []
    aa = []
    length_t32_set = len(t32_set)
    i = 0
    # for i in range(len(t32_set)):
    while i < length_t32_set:
        t = t32_set[i]
        flag = True  # 本次划分是否合理
        if t != 0:
            if len(t) == 4:
                gang.append(t)
            if len(t) == 3:

                if t[0] == t[1]:
                    kz.append(t)
                else:
                    sz.append(t)  # print (sub)
            elif len(t) == 2:
                if t[1] == t[0]:
                    aa.append(t)
                else:
                    t2N.append(t)

        else:
            '修改，使计算时间缩短'
            leftCards = t32_set[i + 1]
            efc_cards = get_effective_cards_gang(dz_set=t2N)  # t2N中不包含ａａ
            # 去除划分不合理的情况，例如345　划分为34　或35等，对于333 划分为33　和3的情况，考虑有将牌的情况暂时不做处理
            for card in leftCards:
                if card in efc_cards:
                    flag = False
                    break

            if flag:
                all.append([gang, kz, sz, aa, t2N, 0, leftCards])
            gang = []
            kz = []
            sz = []
            aa = []
            t2N = []
            i += 1
        i += 1

    allSort = []  # 给每一个元素排序
    allDeWeight = []  # 排序去重后

    for e in all:
        for f in e:
            if f == 0:  # 0是xts位，int不能排序
                continue
            else:
                f.sort()
        allSort.append(e)

    for a in allSort:
        if a not in allDeWeight:
            allDeWeight.append(a)

    allDeWeight = sorted(allDeWeight, key=lambda k: (len(k[0]), len(k[1]), len(k[2])), reverse=True)  # 居然可以这样排序！！
    return allDeWeight

