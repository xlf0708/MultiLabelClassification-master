# ！/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     : 2021/10/21 15:41
# @Author   : Zou
# @Email    : 1091274580@qq.com
# @File     : sichuanMJ_v1.py
# @Software : PyCharm

'''
time: 2022/1/20
author: zou
修复了番型计算中的碰碰胡和金钩钩的判断，以及副露中存在杠的情况
'''

import copy
import os
import time
from ..SCMJ import lib_MJ as MJ
import logging
import datetime
import itertools
if os.path.exists( 'log' ) == False: #
    os.makedirs( 'log' )
#日志输出
logger = logging.getLogger("sichuanMJ_log_v2")
logger.setLevel(level=logging.DEBUG)
time_now = datetime.datetime.now()

handler = logging.FileHandler("log/sichuanMJ_log_v2_%i%i%i.txt" % (time_now.year, time_now.month, time_now.day))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("sichuanMJ_v3 compile finished...")

TIME_START = time.time()
w_type = 0  # lib_MJ的权重选择
ROUND = 0  # 轮数
t3Set = MJ.get_t3info()
t2Set, t2Efc, efc_t2index = MJ.get_t2info()
REMAIN_NUM = 136  # 剩余牌数


T_SELFMO = [0] * 34  # 自摸概率表，牌存在于牌墙中的概率表
LEFT_NUM = [0] * 34  # 未出现的牌的数量表
RT1 = [[0] * 34, [0] * 34]  # 危险度表
RT2 = [[0] * 34, [0] * 34]
RT3 = [[0] * 34, [0] * 34]

t1tot3_dict = MJ.t1tot3_info()
t1tot2_dict = MJ.t1tot2_info()
t2tot3_dict = MJ.t2tot3_info()

class SwitchTiles:
    def __init__(self, hand, n=3):
        """

        :param hand: 手牌
        :param n: 换牌张数，默认为换3张
        """
        self.hand = hand
        self.type = n
        self.color = MJ.splitColor(hand)
        # print(self.color)

    def c_num(self,i):
        """
        换三张使用
        """
        return i % 16
    def judge_cs_value(self,cs):

        """
        输入一个cs(包含杠牌)，返回其最大组合评估值,不包含手牌基础分
        """
        value = 0
        for gang in cs[0]:  # 杠牌
            value += 50
        for aaa in cs[1]:  # kezi
            if self.c_num(aaa[0]) in [1, 2, 8, 9]:
                value += 25
            else:
                value += 18
        for abc in cs[2]:
            value += 10
        for aa in cs[3]:
            if self.c_num(aa[0]) in [1, 2, 8, 9]:
                value += 6
            else:
                value += 5
        for ab in cs[4]:  # 搭子这里需要分4次判断
            if ab[0] + 1 == ab[1]:
                if self.c_num(ab[0]) in [1, 8]:  # 12,89
                    value -= 1
                else:
                    value += 4
            else:
                if self.c_num(ab[0]) in [3, 4, 5]:  # 35、46、57
                    value += 2
                else:
                    value += 1
        for a in cs[-1]:  # 孤张
            if self.c_num(a) in [1, 9]:
                value -= 2

            if self.c_num(a) in [2, 8]:
                value -= 1
        return value

    def choose_color_before(self):
        """
        换三张开始前选择大于3张的最拉胯花色
        """
        len_color = [len(self.color[0]), len(self.color[1]), len(self.color[2])]
        color_3 = []
        for i in range(3):  # 获取大于三的花色
            if len_color[i] >= 3:
                color_3.append(i)
        # print("大于3张花色",color_3)
        min_value = 1000
        min_value_index = -1
        for i in color_3:
            color_cards = self.color[i]
            all_cs = MJ.tree_expand_gang(color_cards)
            one_max = 0    # 记载每种花色的最大值
            for cs in all_cs:
                value_cs = self.judge_cs_value(cs) + len_color[i]*10
                # print(i,cs,value_cs)
                one_max = max(one_max,value_cs)
            # print(one_max)
            if one_max < min_value:
                min_value = one_max
                min_value_index = i
        return min_value_index

    def choose_color_final(self):
        """
        换三张开始后选择大于3张的最拉胯花色
        """
        len_color = [len(self.color[0]), len(self.color[1]), len(self.color[2])]
        color_3 = []
        for i in range(3):  # 加入所有花色
            color_3.append(i)
        min_value = 1000
        min_value_index = -1
        for i in color_3:
            color_cards = self.color[i]
            all_cs = MJ.tree_expand_gang(color_cards)
            one_max = 0    # 记载每种花色的最大值
            for cs in all_cs:
                value_cs = self.judge_cs_value(cs) + len_color[i]*10
                # print(i,cs,value_cs)
                one_max = max(one_max,value_cs)
            # print(one_max)
            if one_max < min_value:
                min_value = one_max
                min_value_index = i
        return min_value_index

    def choose_3card(self):
        """
        从选择好的最拉胯花色选择3张最拉胯的牌
        """
        choose_color = self.choose_color_before()
        # print("color_before",choose_color)
        choose_c_cards = self.color[choose_color]
        gap = -1000
        for cards_3 in itertools.combinations(choose_c_cards, 3):  # 从所选择的花色中取出3个
            cards_3 = list(cards_3)
            cards_other = list(choose_c_cards)
            for card in cards_3:
                cards_other.remove(card)
            cards_3_cs = MJ.tree_expand_gang(cards_3)
            one_max3 = 0  # 记载组合的最大值
            for cs in cards_3_cs:
                value_cs = self.judge_cs_value(cs) + 30
                if value_cs >= one_max3:
                    one_max3 = value_cs
                    max_cs_3 = cs  # 最大权值对应组合
            # print(one_max3, max_cs_3)
            one_max_o = 0
            cards_other_cs = MJ.tree_expand_gang(cards_other)
            for cs in cards_other_cs:
                value_cs = self.judge_cs_value(cs) + 10*len(cards_other)
                if value_cs >= one_max_o:
                    one_max_o = value_cs
                    max_cs_o = cs
            # print(one_max_o, max_cs_o)

            gap_tmp = one_max_o - one_max3
            # print('gap',gap_tmp)
            if gap_tmp > gap:
                gap = gap_tmp
                choose_3cards = cards_3
        # print('-----------------------------------f_gap',gap)
        return choose_3cards


class Node_Qingyise:
    def __init__(self, take=None, AAA=[], ABC=[], T2=[], T1=[], jiang=[], useless_cards=[], raw=[], taking_set=[], taking_set_w=[]):
        self.take = take
        self.AAA = AAA
        self.ABC = ABC
        self.T2 = T2
        self.T1 = T1
        self.jiang = jiang
        self.useless_cards = useless_cards  # 清一色中未用到的花色，优先出牌
        self.raw = raw
        self.xts = 14
        self.taking_set = taking_set
        self.taking_set_w = taking_set_w  # 刻子的权重为6，顺子的权重为为2
        self.ting_info = []
        self.children = []
        self.path_value = None
        self.fan_value = None

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print("AAA:", self.AAA, "ABC:", self.ABC,"jiang:", self.jiang, "T2:",  self.T2, "T1:", self.T1, "raw:", self.raw,
              "taking_set:", self.taking_set, "\nuseless_cards:", self.useless_cards, "ting_info:", self.ting_info,
              "path_value:", self.path_value, "fan_value:", self.fan_value, "xts:", self.xts)

class Qingyise:
    def __init__(self, cards, suits, padding=[]):
        """
        类变量初始化
        :param cards: 手牌
        :param suits: 副露
        :param padding: 填充牌
        """
        self.cards = cards
        self.suits = suits
        self.padding = padding
        self.tree_dict = []
        self.ting_info = []
        self.combination_sets = []
        self.discard_score = {}
        self.discard_state = {}
        self.node_num = 0
        self.chang_num = 0
        self.type_evaluation = 0

    def qingyise_CS(self):
        """
        功能：计算手牌清一色组合
        :return: [[kz], [sz], [aa], [t2], xts, [t1], [useless_cards]]]
        """
        cards = copy.copy(self.cards)
        suits = MJ.deepcopy(self.suits)
        if suits:
            color = suits[0][0] & 0xF0
            for suit in suits:
                if suit[0] & 0xF0 != color:
                    return [[[], [], [], [], 14, [], []]]
        if suits:
            color = suits[0][0] & 0xF0
            useless_cards = []
            for card in cards:
                if card & 0xF0 != color:
                    useless_cards.append(card)
            for card in useless_cards:
                cards.remove(card)
            QYS = MJ.PingHu(cards=cards, suits=suits, padding=self.padding, fan_type=1, useless_cards=useless_cards)
            CS_QYS = QYS.pinghu_CS()
            self.combination_sets = CS_QYS
            return CS_QYS
        color1, color2 = [], []
        for card in cards:
            if card & 0xF0 == cards[0] & 0xF0:
                color1.append(card)
            else:
                color2.append(card)
        if abs(len(color1) - len(color2)) <= 1:
            QYS1 = MJ.PingHu(cards=color1, suits=suits, padding=self.padding, fan_type=1, useless_cards=color2)
            QYS2 = MJ.PingHu(cards=color2, suits=suits, padding=self.padding, fan_type=1, useless_cards=color1)
            CS = QYS1.pinghu_CS() + QYS2.pinghu_CS()
            CS_QYS = sorted(CS, key=lambda k: k[-3])
            # 获取向听数最小的all分支
            min_index = 0
            for i in range(len(CS_QYS)):
                if CS_QYS[i][4] > CS_QYS[0][4]:  # xts+1以下的组合
                    min_index = i
                    break
            if min_index == 0:  # 如果全部都匹配，则min_index没有被赋值，将min_index赋予all长度
                min_index = len(CS_QYS)
            CS_QYS = CS_QYS[:min_index]
            self.combination_sets = CS_QYS
        elif len(color1) < len(color2):
            useless_cards = color1
            cards = color2
            QYS = MJ.PingHu(cards=cards, suits=suits, padding=self.padding, fan_type=1, useless_cards=useless_cards)
            CS_QYS = QYS.pinghu_CS()
            self.combination_sets = CS_QYS
        else:
            useless_cards = color2
            cards = color1
            QYS = MJ.PingHu(cards=cards, suits=suits, padding=self.padding, fan_type=1, useless_cards=useless_cards)
            CS_QYS = QYS.pinghu_CS()
            self.combination_sets = CS_QYS
        return CS_QYS

    def cal_xts(self, node: Node_Qingyise):
        """
         功能：计算节点的向听数
        思路：初始向听数为14，减去相应已成型的组合（kz,sz为３，aa/ab为２），当２Ｎ过剩时，只减去还需要的２Ｎ，对２Ｎ不足时，对还缺少的３Ｎ减去１，表示从孤张牌中选择一张作为３Ｎ的待选
        :param all: [[]]组合信息
        :param suits: 副露
        :return: all　计算向听数后的组合信息
        """
        t3N = node.AAA + node.ABC
        xts = 14 - len(t3N) * 3
        if node.jiang:
            if len(t3N) + len(node.T2) >= 4:
                xts -= (4 - len(t3N)) * 2 + 2
            else:
                xts -= (len(node.T2)) * 2 + 2 + 4 - (len(t3N) + len(node.T2))
        else:
            if len(t3N) + len(node.T2) >= 4:
                xts -= (4 - len(t3N)) * 2 + 1
            else:
                xts -= len(node.T2) * 2 + 1 + 4 - (len(t3N) + len(node.T2))
        cards_num = (len(node.AAA) + len(node.ABC)) * 3 + (len(node.jiang) + len(node.T2)) * 2 + len(node.T1)
        if 14 - cards_num > xts:
            return 14 - cards_num
        return xts

    def fan(self, node, ting_card, cs=[]):
        """
        功能：计算摸到所听牌时的番数
        :param node:
        :param ting_card:
        :param cs:
        :return:
        """
        AAA = MJ.deepcopy(node.AAA)
        ABC = MJ.deepcopy(node.ABC)
        jiang = copy.copy(node.jiang)
        if jiang:
            T3 = cs + [ting_card]
            if T3[0] == T3[1]:
                AAA.append(T3)
            else:
                T3.sort()
                ABC.append(T3)
        else:
            jiang = [ting_card, ting_card]
        fan = 4
        if len(AAA) == 4 and node.jiang:
            fan *= 2
        if len(AAA) == 4 and len(self.suits) == 4:  # 金钩钩
            fan *= 2
        yaojiu = [1, 9, 0x11, 0x19, 0x21, 0x29]
        flag_yaojiu = True
        flag_duanyaojiu = True
        for kz in AAA:
            if kz[0] not in yaojiu:
                flag_yaojiu = False
                break
        for sz in ABC:
            if sz[0] not in yaojiu and sz[2] not in yaojiu:
                flag_yaojiu = False
                break
        if jiang[0] not in yaojiu:
            flag_yaojiu = False
        if flag_yaojiu:
            fan *= 4
        for kz in AAA:
            if kz[0] in yaojiu:
                flag_duanyaojiu = False
                break
        for sz in ABC:
            if sz[0] in yaojiu or sz[2] in yaojiu:
                flag_duanyaojiu = False
                break
        if jiang[0] in yaojiu:
            flag_duanyaojiu = False
        if flag_duanyaojiu:
            fan *= 2
        for suit in self.suits:
            if len(suit) == 4:
                fan *= 2
        return fan

    def ting_module(self, node):
        """
        功能：听牌模块，计算当前节点所听牌及其剩余牌数，番型
        :param node:
        :return:  [ {ting_card: card, remain_num: num, fan: int}, {ting_card: card, remain_num: num, fan: int}, ...]
        """
        ting_info = []
        if node.jiang:  # 将牌存在
            for t2 in node.T2:
                ting_cards = MJ.get_effective_cards(t2)
                for ting_card in ting_cards:
                    remain_num = LEFT_NUM[MJ.convert_hex2index(ting_card)]
                    fan = self.fan(node, ting_card, t2)
                    ting = {'ting_card': ting_card, 'cs': t2, 'remain_num': remain_num, 'fan': fan}
                    ting_info.append(ting)
        else:
            ting_cards = copy.copy(node.T1)
            for t2 in node.T2:
                if t2[0] == t2[1]:
                    node.ting_info = ting_info
                    return
                ting_cards += t2
            for ting_card in ting_cards:
                remain_num = LEFT_NUM[MJ.convert_hex2index(ting_card)]
                fan = self.fan(node, ting_card)
                ting = {'ting_card': ting_card, 'cs': [ting_card], 'remain_num': remain_num, 'fan': fan}
                ting_info.append(ting)
        # node.node_info()
        # print(ting_info)
        node.ting_info = ting_info

    def cal_score(self, node: Node_Qingyise):
        """
        功能：计算牌型评估值
        :param node:
        :return:
        """
        path_value = cal_path_value(copy.copy(node.taking_set), copy.copy(node.taking_set_w))
        if path_value:
            ting_evaluate = 0
            for info in node.ting_info:
                if info['remain_num']:
                    ting_evaluate += info['remain_num'] * info['fan']
            fan_value = path_value * ting_evaluate
            node.path_value = path_value
            node.fan_value = fan_value
        else:
            node.path_value = 0
            node.fan_value = 0

    def expand_node(self, node: Node_Qingyise):
        """
        功能：平胡搜索树节点扩展，
        扩展策略：定将、补刻子、顺子同时进行
        :param node:
        :return:
        """
        # 胡牌判断
        if not node.raw and self.cal_xts(node) == 1:
            node.xts = 1
            self.node_num += 1
            self.ting_module(node=node)
            self.cal_score(node=node)
            if node.ting_info:
                for info in node.ting_info:
                    info['score'] = node.path_value * info['remain_num'] * info['fan']
            return

        has_jiang = False
        if node.jiang == [] and not node.raw:  # 将牌的扩展，用对子或用孤张牌
            if len(node.ABC) + len(node.AAA) + len(node.T2) > 4:
                for t2 in node.T2:
                    if t2[0] == t2[1]:
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        child = Node_Qingyise(take=-1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2,
                                              T1=copy.copy(node.T1), jiang=t2, raw=MJ.deepcopy(node.raw),
                                              taking_set=copy.copy(node.taking_set),
                                              taking_set_w=copy.copy(node.taking_set_w),
                                              useless_cards=copy.copy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                        has_jiang = True

            t2_jiang = False
            if len(node.ABC) + len(node.AAA) + len(node.T2) <= 4:
                for t2 in node.T2:
                    if t2[0] == t2[1]:
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        child = Node_Qingyise(take=-1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2,
                                              T1=copy.copy(node.T1), jiang=t2, taking_set=copy.copy(node.taking_set),
                                              taking_set_w=copy.copy(node.taking_set_w),
                                              useless_cards=copy.copy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                        t2_jiang = True

            if not has_jiang and not t2_jiang:
                jiangs = copy.copy(node.T1)
                if not jiangs:
                    for t2 in node.T2:
                        jiangs = t2
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        for t1 in jiangs:
                            taking_set = copy.copy(node.taking_set)
                            taking_set.append(t1)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set_w.append(1)
                            T1 = copy.copy(jiangs)
                            T1.remove(t1)
                            child = Node_Qingyise(take=t1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                                  jiang=[t1, t1], T2=T2, T1=T1, taking_set=taking_set,
                                                  taking_set_w=taking_set_w,
                                                  useless_cards=node.useless_cards)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                else:
                    for t1 in jiangs:
                        if t1 == -1:  # 对-1不作扩展
                            continue
                        taking_set = copy.copy(node.taking_set)
                        taking_set.append(t1)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.append(1)
                        T1 = copy.copy(jiangs)
                        T1.remove(t1)
                        child = Node_Qingyise(take=t1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                              jiang=[t1, t1], T2=MJ.deepcopy(node.T2), T1=T1, taking_set=taking_set,
                                              taking_set_w=taking_set_w,
                                              useless_cards=node.useless_cards)
                        node.add_child(child=child)
                        self.expand_node(node=child)

        # T3扩展
        if len(node.AAA) + len(node.ABC) != 4 and not has_jiang:
            # 当待扩展集合不为空时，使用该集合进行扩展
            if node.raw:
                tn = node.raw[-1]
                raw = MJ.deepcopy(node.raw)  # 深度搜索后面的节点会改变raw，回退可能导致前面的节点raw不正确，这里需要copy
                raw.pop()
                if type(tn) == list:  # 使用t2扩展t3
                    t2 = tn
                    for item in t2tot3_dict[str(t2)]:  # "t2": [[t2_,t3,t1_left,valid,p]]
                        AAA = MJ.deepcopy(node.AAA)
                        ABC = MJ.deepcopy(node.ABC)
                        if item[1][0] == item[1][1]:
                            AAA.append(item[1])
                        else:
                            ABC.append(item[1])
                        taking_set = copy.copy(node.taking_set)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set.append(item[-2])
                        taking_set_w.append(item[-1])
                        child = Node_Qingyise(take=item[-2], AAA=AAA, ABC=ABC, jiang=copy.copy(node.jiang),
                                              T2=MJ.deepcopy(node.T2), T1=copy.copy(node.T1),
                                              raw=raw, taking_set=taking_set, taking_set_w=taking_set_w,
                                              useless_cards=node.useless_cards)
                        node.add_child(child=child)
                        self.expand_node(node=child)
                elif type(tn) == int:
                    t1 = tn
                    for item in t1tot3_dict[str(t1)]:  # {"t1":[[t3,t2(valid card),p]]}
                        AAA = MJ.deepcopy(node.AAA)
                        ABC = MJ.deepcopy(node.ABC)
                        if item[0][0] == item[0][1]:
                            AAA.append(item[0])
                        else:
                            ABC.append(item[0])
                        take = item[1]
                        take_w = item[-1]
                        taking_set = copy.copy(node.taking_set)
                        taking_set.extend(take)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.extend(take_w)
                        child = Node_Qingyise(take=take, AAA=AAA, ABC=ABC, jiang=copy.copy(node.jiang),
                                              T2=MJ.deepcopy(node.T2), T1=copy.copy(node.T1), raw=raw,
                                              taking_set=taking_set, taking_set_w=taking_set_w,
                                              useless_cards=node.useless_cards)
                        node.add_child(child=child)
                        self.expand_node(node=child)
                else:
                    print("tn Error")
            else:
                t3_num = 3 if node.jiang else 4  # 如果已经定将，只需要凑齐3个刻子，顺子
                if node.T2:  # 1、先扩展T2为T3
                    t2_sets = MJ.deepcopy(node.T2)
                    for t2_set in itertools.combinations(t2_sets, min(t3_num - len(node.AAA) - len(node.ABC), len(t2_sets))):
                        if t2_set:
                            T2 = MJ.deepcopy(node.T2)
                            for t2 in t2_set:
                                T2.remove(t2)
                            child = Node_Qingyise(AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                                  jiang=copy.copy(node.jiang), T2=T2, T1=copy.copy(node.T1),
                                                  raw=list(t2_set), taking_set=copy.copy(node.taking_set),
                                                  taking_set_w=copy.copy(node.taking_set_w),
                                                  useless_cards=node.useless_cards)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                elif node.T1:
                    t1_sets = copy.copy(node.T1)
                    #这里移除了填充的-1，不作扩展
                    if -1 in t1_sets:
                        t1_sets.remove(-1)
                    for t1_set in itertools.combinations(t1_sets, min(t3_num - len(node.AAA) - len(node.ABC), len(t1_sets))):
                        if t1_set:
                            T1 = copy.copy(node.T1)
                            for t1 in t1_set:
                                T1.remove(t1)
                            child = Node_Qingyise(AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                                  jiang=copy.copy(node.jiang), T2=MJ.deepcopy(node.T2), T1=T1,
                                                  raw=list(t1_set), taking_set=copy.copy(node.taking_set),
                                                  taking_set_w=copy.copy(node.taking_set_w),
                                                  useless_cards=node.useless_cards)
                            node.add_child(child=child)
                            self.expand_node(node=child)

        # 定将完之后，只有3个刻子、顺子再加孤张牌的情况，用一张孤张牌扩展成T2使xts==1
        if self.cal_xts(node) == 2 and len(node.AAA) + len(node.ABC) == 3 and node.jiang and not node.raw:
            for t1 in node.T1:
                for item in t1tot2_dict[str(t1)]:  # "t1": [t2, t1_left, p]
                    T2 = MJ.deepcopy(node.T2)
                    T2.append(item[0])
                    T1 = copy.copy(node.T1)
                    T1.remove(t1)
                    taking_set = copy.copy(node.taking_set)
                    taking_set_w = copy.copy(node.taking_set_w)
                    taking_set.append(item[1])
                    taking_set_w.append(item[-1])
                    child = Node_Qingyise(take=item[1], AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                          jiang=MJ.deepcopy(node.jiang), T2=T2, T1=T1,
                                          taking_set=taking_set, taking_set_w=taking_set_w,
                                          useless_cards=node.useless_cards)
                    node.add_child(child=child)
                    self.expand_node(node=child)

        if self.cal_xts(node) > 1 and not node.T2 and not node.T1 and not node.raw:  # 手牌中的牌不够用时，需要再加一张牌
            taking_set = copy.copy(node.taking_set)
            taking_set_w = copy.copy(node.taking_set_w)
            left_num = copy.copy(LEFT_NUM)
            for take in taking_set:
                left_num[MJ.convert_hex2index(take)] -= 1
            t3 = node.AAA + node.ABC
            color = t3[0][0] & 0xF0
            for i in range(1, 10):
                card = color + i
                if left_num[MJ.convert_hex2index(card)]:
                    child = Node_Qingyise(take=card, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                          jiang=MJ.deepcopy(node.jiang), T1=[card],
                                          taking_set=taking_set + [card],
                                          taking_set_w=taking_set_w + [1], useless_cards=node.useless_cards)
                    node.add_child(child=child)
                    self.expand_node(node=child)

    def generate_tree(self):
        """
        功能：生成搜索树
        :return:
        """
        kz = []
        sz = []
        for t3 in self.suits:
            if t3[0] == t3[1]:
                kz.append([t3[0], t3[0], t3[0]])
            else:
                sz.append(t3)
        CS = self.combination_sets
        for cs in CS:
            root = Node_Qingyise(take=None, AAA=cs[0]+kz, ABC=cs[1]+sz, T2=cs[2]+cs[3], T1=cs[5], useless_cards=cs[-1])
            self.tree_dict.append(root)
            self.expand_node(node=root)
            #traverse(root)

    def cards_type_evaluation(self, node: Node_Qingyise):
        if node.xts == 1:
            self.type_evaluation += node.fan_value
        elif node.children:
            for child in node.children:
                self.cards_type_evaluation(node=child)

    def get_fan_score(self):
        self.generate_tree()
        for root in self.tree_dict:
            self.cards_type_evaluation(root)
        return self.type_evaluation

    def calculate_path_expectation(self, node):
        #深度搜索
        if node.ting_info and node.xts == 1:
            self.node_num += 1
            discard_set = []
            for t2 in node.T2:
                discard_set.extend(t2)
            discard_set.extend(node.T1)
            discard_set.extend(node.useless_cards)
            taking_set_sorted = sorted(node.taking_set)
            taking_set_lable = str(taking_set_sorted)  # 转化为str可以加快查找
            if discard_set == []:
                logger.info("qingyise_error:AAA %s, ABC %s, jiang:%s, T2:%s, T1:%s", node.AAA, node.ABC, node.jiang, node.T2, node.T1 )
                return
            # todo 这种按摸牌的评估方式是否唯一准确
            for card in list(set(discard_set)):
                score = 0
                for info in node.ting_info:
                    if card not in info['cs']:
                        score += info['score']
                if card not in self.discard_state.keys():
                    self.discard_state[card] = [[], []]
                if taking_set_lable not in self.discard_state[card][0]:
                    self.discard_state[card][0].append(taking_set_lable)
                    self.discard_state[card][-1].append(score)
                else:
                    index = self.discard_state[card][0].index(taking_set_lable)
                    if score > self.discard_state[card][-1][index]:
                        self.chang_num += 1
                        self.discard_state[card][-1][index] = score

        elif node.children != []:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):
        for root in self.tree_dict:
            self.calculate_path_expectation(root)
        state_num = 0
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][-1])
            state_num += len(self.discard_state[discard][-1])

        return self.discard_score

class Node_Pengpenghu:
    def __init__(self, take=None, AAA=[], AA=[], T1=[], raw=[], taking_set=[], taking_set_w=[]):
        self.take = take
        self.AAA = AAA
        self.AA = AA
        self.T1 = T1
        self.raw = raw
        self.xts = 14  # 向听数为1作为扩展结束的标志
        self.taking_set = taking_set
        self.taking_set_w = taking_set_w  # 刻子的权重为6， 将牌的权重为2， 孤张牌扩展为2
        self.ting_info = []
        self.children = []
        self.ting_info = []
        self.path_value = None
        self.fan_value = None

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print("AAA:", self.AAA, "AA:", self.AA, "T1:", self.T1, "raw:", self.raw,
              "taking_set:", self.taking_set, "\nting_info:", self.ting_info,
              "path_value:", self.path_value, "fan_value:", self.fan_value, "xts:", self.xts)

class Pengpenghu:
    def __init__(self, cards=[], suits=[], padding=[]):
        """
        类变量初始化
        :param cards: 手牌
        :param suits: 副露
        :param padding: 填充牌
        """
        self.cards = cards
        self.suits = suits
        self.padding = padding
        self.combination_sets = []
        self.tree_dict = []
        self.discard_score = {}
        self.discard_state = {}
        self.node_num = 0
        self.chang_num = 0
        self.type_evaluation = 0

    def calculate_xts_pengpenghu(self, AAA=[], AA=[]):
        """
        功能：计算碰碰胡组合的xts
        :param AAA: 手牌组合中的刻子
        :param AA: 手牌中的对子
        :return: 返回手牌的xts
        """
        xts = 14 - len(AAA) * 3
        if len(AAA) + len(AA) >= 5:
            xts -= (5 - len(AAA)) * 2
        elif (len(AAA) + len(AA)) >= 4:
            xts -= (4 - len(AAA)) * 2 + 1
        else:
            xts -= len(AA) * 2 + (4 - len(AAA) - len(AA)) + 1
        return xts

    def pengpenghu_CS(self):
        """
        功能：输入手牌，计算碰碰胡的手牌组合
        :return: CS = [[刻子], [对子], [孤张牌], 向听数]]
        """
        CS = [[], [], [], 14]
        for suit in self.suits:
            if suit[0] != suit[1]:
                return CS
        for suit in self.suits:
            CS[0].append([suit[0], suit[0], suit[0]])
        cards_set = set(self.cards)
        for card in cards_set:
            if self.cards.count(card) == 4:
                CS[0].append([card, card, card])
                CS[2].append(card)
            elif self.cards.count(card) == 3:
                CS[0].append([card, card, card])
            elif self.cards.count(card) == 2:
                CS[1].append([card, card])
            else:
                CS[2].append(card)
        CS[-1] = self.calculate_xts_pengpenghu(CS[0], CS[1])
        self.combination_sets = CS
        return CS

    def fan(self, node, ting_card):
        """
        功能：计算达到胡牌状态时手牌的番型
        :param node:
        :param ting_card:
        :return:
        """
        fan = 2
        if len(self.suits) == 4:  # 金钩钩
            fan *= 2
        if len(node.AAA) == 4:
            KZs = MJ.deepcopy(node.AAA)
            jiangs = [[ting_card, ting_card]]
        else:
            KZs = MJ.deepcopy(node.AAA) + [[ting_card, ting_card, ting_card]]
            AAs = MJ.deepcopy(node.AA)
            AAs.remove([ting_card, ting_card])
            jiangs = AAs
        flag_qingyise = True
        flag_yaojiu = True
        flag_duanyaojiu = True
        # qingyise
        color = KZs[0][0] & 0xF0
        yaojiu = [1, 9, 0x11, 0x19, 0x21, 0x29]
        for kz in KZs:
            if kz[0] & 0xF0 != color:
                flag_qingyise = False
                break
        for jiang in jiangs:
            if jiang[0] & 0xF0 != color:
                flag_qingyise = False

        # yaojiu、duanyaojiu
        for kz in KZs:
            if kz[0] not in yaojiu:
                flag_yaojiu = False
                break
        for jiang in jiangs:
            if jiang[0] not in yaojiu:
                flag_yaojiu = False
        for kz in KZs:
            if kz[0] in yaojiu:
                flag_duanyaojiu = False
        for jiang in jiangs:
            if jiang[0] in yaojiu:
                flag_duanyaojiu = False
        if flag_qingyise:
            fan *= 2
        if flag_yaojiu:
            fan *= 4
        if flag_duanyaojiu:
            fan *= 2
        for suit in self.suits:
            if len(suit) == 4:
                fan *= 2
        return fan

    def ting_module(self, node):
        """
        功能：计算手牌达到碰碰胡听牌阶段听哪些牌，剩余牌、番数
        :param node:
        :return: [{'ting_card': card, 'remain_num': num, 'fan': num, 'score': score}, ...]
        """
        ting_info = []
        if len(node.AAA) == 4:
            for ting_card in node.T1:
                remain_num = LEFT_NUM[MJ.convert_hex2index(ting_card)]
                fan = self.fan(node, ting_card)
                ting = {'ting_card': ting_card, 'remain_num': remain_num, 'fan': fan}
                ting_info.append(ting)
        elif len(node.AAA) == 3:
            for AA in node.AA:
                remain_num = LEFT_NUM[MJ.convert_hex2index(AA[0])]
                fan = self.fan(node, AA[0])
                ting = {'ting_card': AA[0], 'remain_num': remain_num, 'fan': fan}
                ting_info.append(ting)
        node.ting_info = ting_info

    def cal_score(self, node: Node_Pengpenghu):
        """
        功能: 计算评估值
        :param node:
        :return:
        """
        path_value = cal_path_value(copy.copy(node.taking_set), copy.copy(node.taking_set_w))
        if path_value:
            ting_evaluate = 0
            for info in node.ting_info:
                if info['remain_num']:
                    ting_evaluate += info['remain_num'] * info['fan']
            fan_value = path_value * ting_evaluate
            node.path_value = path_value
            node.fan_value = fan_value
        else:
            node.path_value = 0
            node.fan_value = 0

    def expand_node(self, node: Node_Pengpenghu):
        """
        功能：碰碰胡牌型的搜索树节点扩展
        扩展策略：
        1、若待扩展集合为空，从T1集合中选择 5 - len(node.AAA) - len(node.AA)张牌加入到raw中， 将raw中的孤张牌扩展为AA或AAA，
        使手牌达到 3(AAA) + 2(AA)的形式。
        或者从T1集合中选择4 - len(node.AAA) - len(node.AA)张牌加入到raw中，扩展完成之后手牌达到 4(AAA) 的形式。
        2、待扩展集合不为空，若len(node.AAA) + len(node.AA) + len(node.raw) == 5 , 则扩展至3(AAA) + 2(AA) 的形式，
        若len(node.AAA) + len(node.AA) + len(node.raw) == 4， 则扩展至4(AAA)的形式。
        :param node: 带扩展节点
        :return:
        """
        if not node.raw and self.calculate_xts_pengpenghu(AAA=node.AAA, AA=node.AA) == 1:
            node.xts = 1
            self.node_num += 1
            self.ting_module(node=node)
            self.cal_score(node=node)
            if node.ting_info:
                for info in node.ting_info:
                    info['score'] = node.path_value * info['remain_num'] * info['fan']
            return
        else:
            if node.raw:  # 待扩展集不为空
                tn = node.raw[-1]
                if type(tn) == int:  # 孤张牌的扩展
                    card = tn
                    raw = MJ.deepcopy(node.raw)
                    raw.pop()
                    AA = MJ.deepcopy(node.AA)
                    AA.append([card, card])
                    taking_set = copy.copy(node.taking_set)
                    taking_set.append(card)
                    taking_set_w = copy.copy(node.taking_set_w)
                    taking_set_w.append(1)
                    child = Node_Pengpenghu(take=card, AAA=MJ.deepcopy(node.AAA), AA=AA, T1=copy.copy(node.T1),
                                            raw=raw, taking_set=taking_set, taking_set_w=taking_set_w)
                    node.add_child(child=child)
                    self.expand_node(node=child)
                else:  # 对子扩展成刻子
                    card = tn[0]
                    raw = MJ.deepcopy(node.raw)
                    raw.pop()
                    AAA = MJ.deepcopy(node.AAA)
                    AAA.append([card, card, card])
                    taking_set = copy.copy(node.taking_set)
                    taking_set.append(card)
                    taking_set_w = copy.copy(node.taking_set_w)
                    taking_set_w.append(6)  # todo 这个地方会出现偏差，如果是有孤张牌扩展成对子，再扩展成刻子那权重应该还是1
                    child = Node_Pengpenghu(take=card, AAA=AAA, AA=MJ.deepcopy(node.AA), T1=copy.copy(node.T1),
                                            raw=raw, taking_set=taking_set, taking_set_w=taking_set_w)
                    node.add_child(child=child)
                    self.expand_node(node=child)
            else:
                if len(node.AAA) + len(node.AA) >= 5:  # 扩展成 3(AAA) + 2(AA) 的形式
                    AA_sets = node.AA
                    for AA_set in itertools.combinations(AA_sets, min(3 - len(node.AAA), len(AA_sets))):
                        AA = MJ.deepcopy(node.AA)
                        raw = list(AA_set)
                        for aa in raw:
                            AA.remove(aa)
                        child = Node_Pengpenghu(take=-1, AAA=MJ.deepcopy(node.AAA), AA=AA, T1=copy.copy(node.T1),
                                                raw=raw, taking_set=copy.copy(node.taking_set),
                                                taking_set_w=copy.copy(node.taking_set_w))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                elif len(node.AAA) + len(node.AA) == 4:  # 扩展成 4(AAA)的形式
                    AA_sets = node.AA
                    for AA_set in itertools.combinations(AA_sets, min(4 - len(node.AAA), len(AA_sets))):
                        AA = MJ.deepcopy(node.AA)
                        raw = list(AA_set)
                        for aa in raw:
                            AA.remove(aa)
                        child = Node_Pengpenghu(take=-1, AAA=MJ.deepcopy(node.AAA), AA=AA, T1=copy.copy(node.T1),
                                                raw=raw, taking_set=copy.copy(node.taking_set),
                                                taking_set_w=copy.copy(node.taking_set_w))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                elif len(node.T1):
                    t1_sets = copy.copy(node.T1)
                    if -1 in t1_sets:
                        t1_sets.remove(-1)
                    # 扩展成 3(AAA) + 2(AA) 的形式, 此时的xts=1
                    for t1_set in itertools.combinations(t1_sets, min(5 - len(node.AAA) - len(node.AA), len(t1_sets))):
                        if t1_set:
                            T1 = copy.copy(node.T1)
                            raw = list(t1_set)
                            for t1 in t1_set:
                                T1.remove(t1)
                            child = Node_Pengpenghu(take=-1, AAA=MJ.deepcopy(node.AAA), AA=MJ.deepcopy(node.AA), T1=T1,
                                                    raw=raw, taking_set=copy.copy(node.taking_set),
                                                    taking_set_w=copy.copy(node.taking_set_w))
                            node.add_child(child=child)
                            self.expand_node(node=child)
                    # 扩展成4(AAA)的形式, 4(AAA)是xts=1
                    for t1_set in itertools.combinations(t1_sets, min(4 - len(node.AAA) - len(node.AA), len(t1_sets))):
                        T1 = copy.copy(node.T1)
                        raw = list(t1_set)
                        for t1 in t1_set:
                            T1.remove(t1)
                        child = Node_Pengpenghu(take=-1, AAA=MJ.deepcopy(node.AAA), AA=MJ.deepcopy(node.AA), T1=T1,
                                                raw=raw, taking_set=copy.copy(node.taking_set),
                                                taking_set_w=copy.copy(node.taking_set_w))
                        node.add_child(child=child)
                        self.expand_node(node=child)

    def generate_tree(self):
        CS = self.combination_sets
        root = Node_Pengpenghu(take=None, AAA=CS[0], AA=CS[1], T1=CS[2])
        self.tree_dict.append(root)
        self.expand_node(node=root)
        # traverse(root)

    def cards_type_evaluation(self, node: Node_Pengpenghu):
        if node.xts == 1:
            self.type_evaluation += node.fan_value
        elif node.children:
            for child in node.children:
                self.cards_type_evaluation(node=child)

    def get_fan_score(self):
        self.generate_tree()
        for root in self.tree_dict:
            self.cards_type_evaluation(root)
        return self.type_evaluation

    def calculate_path_expectation(self, node: Node_Pengpenghu):
        """
        功能：搜索树的路劲评估，计算该路径的摸牌集合和废牌集合
        :param node:
        :return:
        """
        # 深度搜索
        if node.ting_info and node.xts == 1:
            # 达到听牌阶段
            discard_set = []
            discard_set.extend(node.T1)
            taking_set_sorted = sorted(node.taking_set)
            taking_set_lable = str(taking_set_sorted)
            # 对应的情况为 AAA: [[7, 7, 7], [34, 34, 34], [1, 1, 1]] AA: [[4, 4], [8, 8], [41, 41]] T1: [] raw: [] taking_set: [34, 1]
            if not discard_set:
                for aa in node.AA:
                    discard_set.append(aa[0])
            for card in list(set(discard_set)):
                if card not in self.discard_state.keys():
                    self.discard_state[card] = [[], []]
                score = 0
                for info in node.ting_info:
                    if info['ting_card'] != card:
                        score += info['score']
                if taking_set_lable not in self.discard_state[card][0]:
                    self.discard_state[card][0].append(taking_set_lable)
                    self.discard_state[card][1].append(score)
                else:
                    index = self.discard_state[card][0].index(taking_set_lable)
                    if score > self.discard_state[card][-1][index]:
                        self.chang_num += 1
                        self.discard_state[card][-1][index] = score
        elif node.children:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):
        for root in self.tree_dict:
            self.calculate_path_expectation(root)
        state_num = 0
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][1])
            state_num += len(self.discard_state[discard][-1])
        # print("leaf node ", self.node_num)
        # print("state_num", state_num)
        # print("chang_num", self.chang_num)
        return self.discard_score

class Node_PH:
    def __init__(self, take=None, AAA=[], ABC=[], T2=[], T1=[], jiang=[], raw=[], taking_set=[], taking_set_w=[]):
        self.take = take
        self.AAA = AAA
        self.ABC = ABC
        self.T2 = T2
        self.T1 = T1
        self.jiang = jiang
        self.raw = raw
        self.xts = 14
        self.children = []
        self.taking_set = taking_set
        self.taking_set_w = taking_set_w
        self.ting_info = []
        self.path_value = None
        self.fan_value = None

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print("AAA:", self.AAA, "ABC:", self.ABC, "jiang:", self.jiang, "T2:", self.T2, "T1:", self.T1,
              "taking_set:", self.taking_set, "raw:", self.raw, "\nuseless_cards:",
              "ting_info:", self.ting_info, "path_value:", self.path_value, "fan_value:", self.fan_value, "xts:",
              self.xts)

class SearchTree_PH():
    def __init__(self, cards, suits, padding):
        self.cards = cards
        self.suits = suits
        self.padding = padding
        self.tree_dict = []
        self.ting_sets = []
        self.combination_sets = []
        self.cards_type_score = {}
        self.discard_score = {}
        self.discard_state = {}
        self.type_evaluation = 0
        self.node_num = 0
        self.chang_num = 0

    def pinghu_CS(self, cards=[], suits=[]):
        """
        功能：综合计算手牌的组合信息
        思路：对手牌进行花色分离后，单独计算出每种花色的组合信息　，再将其综合起来，计算每个组合向听数，最后输出最小向听数及其加一的组合
        :param cards: 手牌
        :param suits: 副露
        :return: 组合信息
        """
        PH = MJ.PingHu(cards=copy.copy(self.cards), suits=MJ.deepcopy(self.suits), padding=self.padding, fan_type=0)
        CS_PH = PH.pinghu_CS()
        self.combination_sets = CS_PH
        return CS_PH

    def cal_xts(self, node: Node_PH):
        """
         功能：计算节点的向听数
        思路：初始向听数为14，减去相应已成型的组合（kz,sz为３，aa/ab为２），当２Ｎ过剩时，只减去还需要的２Ｎ，对２Ｎ不足时，对还缺少的３Ｎ减去１，表示从孤张牌中选择一张作为３Ｎ的待选
        :param all: [[]]组合信息
        :param suits: 副露
        :return: all　计算向听数后的组合信息
        """
        t3N = node.AAA + node.ABC
        xts = 14 - len(t3N) * 3
        if node.jiang:
            if len(t3N) + len(node.T2) >= 4:
                xts -= (4 - len(t3N)) * 2 + 2
            else:
                xts -= (len(node.T2)) * 2 + 2 + 4 - (len(t3N) + len(node.T2))
        else:
            if len(t3N) + len(node.T2) >= 4:
                xts -= (4 - len(t3N)) * 2 + 1
            else:
                xts -= len(node.T2) * 2 + 1 + 4 - (len(t3N) + len(node.T2))
        return xts

    def fan(self, node, ting_card, cs=[]):
        """
        功能：计算摸到所听牌时的番数
        :param node:
        :param ting_card:
        :return:
        """
        AAA = MJ.deepcopy(node.AAA)
        ABC = MJ.deepcopy(node.ABC)
        jiang = copy.copy(node.jiang)
        if jiang:
            T3 = cs + [ting_card]
            if T3[0] != T3[1]:
                T3.sort()
                ABC.append(T3)
            else:
                AAA.append(T3)
        else:
            jiang = [ting_card, ting_card]
        fan = 1
        if len(AAA) == 4:  # pengpenghu
            fan *= 2
        elif len(self.suits) == 4:  # jingougou
            fan *= 2
        color = jiang[0] & 0xF0
        flag_qingyise = True
        for T3 in AAA + ABC:
            if T3[0] & 0xF0 != color:
                flag_qingyise = False
                break
        if flag_qingyise:
            fan *= 4
        yaojiu = [1, 9, 0x11, 0x19, 0x21, 0x29]
        flag_yaojiu = True
        flag_duanyaojiu = True
        for kz in AAA:
            if kz[0] not in yaojiu:
                flag_yaojiu = False
                break
        for sz in ABC:
            if sz[0] not in yaojiu and sz[2] not in yaojiu:
                flag_yaojiu = False
                break
        if jiang[0] not in yaojiu:
            flag_yaojiu = False
        if flag_yaojiu:
            fan *= 4
        for kz in AAA:
            if kz[0] in yaojiu:
                flag_duanyaojiu = False
                break
        for sz in ABC:
            if sz[0] in yaojiu or sz[2] in yaojiu:
                flag_duanyaojiu = False
                break
        if jiang[0] in yaojiu:
            flag_duanyaojiu = False
        if flag_duanyaojiu:
            fan *= 2
        for suit in self.suits:
            if len(suit) == 4:
                fan *= 2
        return fan

    def ting_module(self, node):
        """
        功能：听牌模块，计算当前节点所听牌及其剩余牌数，番型
        :param node:
        :return:  [ {ting_card: card, cs, remain_num: num, fan: int}, {ting_card: card, remain_num: num, fan: int}, ...]
        """
        ting_info = []
        if node.jiang:  # 将牌存在
            for t2 in node.T2:
                ting_cards = MJ.get_effective_cards(t2)
                for ting_card in ting_cards:
                    remain_num = LEFT_NUM[MJ.convert_hex2index(ting_card)]
                    fan = self.fan(node, ting_card, t2)
                    ting = {'ting_card': ting_card, 'cs': t2, 'remain_num': remain_num, 'fan': fan}
                    ting_info.append(ting)
        else:
            ting_cards = copy.copy(node.T1)
            for t2 in node.T2:
                if t2[0] == t2[1]:
                    node.ting_info = ting_info
                    return
                else:
                    ting_cards += t2
            for ting_card in ting_cards:
                remain_num = LEFT_NUM[MJ.convert_hex2index(ting_card)]
                fan = self.fan(node, ting_card)
                ting = {'ting_card': ting_card, 'cs': [ting_card], 'remain_num': remain_num, 'fan': fan}
                ting_info.append(ting)
        node.ting_info = ting_info

    def cal_score(self, node: Node_PH):
        path_value = cal_path_value(copy.copy(node.taking_set), copy.copy(node.taking_set_w))
        if path_value:
            ting_evaluate = 0
            for info in node.ting_info:
                if info['remain_num']:
                    ting_evaluate += info['remain_num'] * info['fan']
            fan_value = path_value * ting_evaluate
            node.path_value = path_value
            node.fan_value = fan_value
        else:
            node.path_value = 0
            node.fan_value = 0

    def expand_node(self, node):
        """
        功能：平胡搜索树节点扩展，
        扩展策略：定将、补刻子、顺子同时进行
        :param node:
        :return:
        """
        # 胡牌判断
        if not node.raw and self.cal_xts(node) == 1:
            node.xts = 1
            self.ting_module(node=node)
            self.cal_score(node=node)
            if node.ting_info:
                for info in node.ting_info:
                    info['score'] = node.path_value * info['remain_num'] * info['fan']
            return

        has_jiang = False
        if node.jiang == [] and not node.raw:  # 将牌的扩展，用对子或用孤张牌
            if len(node.ABC) + len(node.AAA) + len(node.T2) > 4:
                for t2 in node.T2:
                    if t2[0] == t2[1]:
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        child = Node_PH(take=-1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2,
                                        T1=copy.copy(node.T1), jiang=t2,
                                        raw=MJ.deepcopy(node.raw), taking_set=copy.copy(node.taking_set),
                                        taking_set_w=copy.copy(node.taking_set_w))
                        node.add_child(child=child)
                        self.expand_node(child)
                        has_jiang = True

            t2_jiang = False
            if len(node.ABC) + len(node.AAA) + len(node.T2) <= 4:
                for t2 in node.T2:
                    if t2[0] == t2[1]:
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        child = Node_PH(take=-1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2,
                                        T1=copy.copy(node.T1), jiang=t2,
                                        raw=MJ.deepcopy(node.raw), taking_set=copy.copy(node.taking_set),
                                        taking_set_w=copy.copy(node.taking_set_w))
                        node.add_child(child=child)
                        self.expand_node(child)
                        t2_jiang = True

            if not has_jiang and not t2_jiang:
                jiangs = copy.copy(node.T1)
                if jiangs == []:
                    for t2 in node.T2:
                        jiangs = t2
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        for t1 in jiangs:
                            taking_set = copy.copy(node.taking_set)
                            taking_set.append(t1)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set_w.append(1)
                            T1 = copy.copy(jiangs)
                            T1.remove(t1)
                            child = Node_PH(take=t1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), jiang=[t1, t1],
                                            T2=T2, T1=T1, taking_set=taking_set, taking_set_w=taking_set_w)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                else:
                    for t1 in jiangs:
                        if t1 == -1:  # 对-1不作扩展
                            continue
                        taking_set = copy.copy(node.taking_set)
                        taking_set.append(t1)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.append(1)
                        T1 = copy.copy(jiangs)
                        T1.remove(t1)
                        child = Node_PH(take=t1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), jiang=[t1, t1],
                                        T2=MJ.deepcopy(node.T2), T1=T1, taking_set=taking_set, taking_set_w=taking_set_w)
                        node.add_child(child=child)
                        self.expand_node(node=child)

        # T3扩展
        if len(node.AAA) + len(node.ABC) != 4 and not has_jiang:
            # 当待扩展集合不为空时，使用该集合进行扩展
            if node.raw:
                tn = node.raw[-1]
                raw = MJ.deepcopy(node.raw)  # 深度搜索后面的节点会改变raw，回退可能导致前面的节点raw不正确，这里需要copy
                raw.pop()
                if type(tn) == list:  # 使用t2扩展t3
                    t2 = tn
                    for item in t2tot3_dict[str(t2)]:  # "t2": [[t2_,t3,t1_left,valid,p]]
                        AAA = MJ.deepcopy(node.AAA)
                        ABC = MJ.deepcopy(node.ABC)
                        if item[1][0] == item[1][1]:
                            AAA.append(item[1])
                        else:
                            ABC.append(item[1])
                        taking_set = copy.copy(node.taking_set)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set.append(item[-2])
                        taking_set_w.append(item[-1])
                        child = Node_PH(take=item[-2], AAA=AAA, ABC=ABC, jiang=copy.copy(node.jiang),
                                        T2=MJ.deepcopy(node.T2), T1=copy.copy(node.T1), raw=raw,
                                        taking_set=taking_set, taking_set_w=taking_set_w)
                        node.add_child(child=child)
                        self.expand_node(node=child)
                elif type(tn) == int:
                    t1 = tn
                    for item in t1tot3_dict[str(t1)]:  # {"t1":[[t3,t2(valid card),p]]}
                        AAA = MJ.deepcopy(node.AAA)
                        ABC = MJ.deepcopy(node.ABC)
                        if item[0][0] == item[0][1]:
                            AAA.append(item[0])
                        else:
                            ABC.append(item[0])
                        take = item[1]
                        take_w = item[-1]
                        taking_set = copy.copy(node.taking_set)
                        taking_set.extend(take)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.extend(take_w)
                        child = Node_PH(take=take, AAA=AAA, ABC=ABC, jiang=copy.copy(node.jiang), T2=MJ.deepcopy(node.T2),
                                        T1=copy.copy(node.T1), raw=raw,
                                        taking_set=taking_set, taking_set_w=taking_set_w)
                        node.add_child(child=child)
                        self.expand_node(node=child)
                else:
                    print("tn Error")
            else:
                t3_num = 3 if node.jiang else 4  # 如果已经定将，只需要凑齐3个刻子，顺子
                if node.T2:  # 1、先扩展T2为T3
                    t2_sets = MJ.deepcopy(node.T2)
                    for t2_set in itertools.combinations(t2_sets, min(t3_num - len(node.AAA) - len(node.ABC), len(t2_sets))):
                        if t2_set:
                            T2 = MJ.deepcopy(node.T2)
                            raw = list(t2_set)
                            for t2 in raw:
                                T2.remove(t2)
                            child = Node_PH(AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2,
                                            T1=copy.copy(node.T1), jiang=copy.copy(node.jiang), raw=raw,
                                            taking_set=copy.copy(node.taking_set), taking_set_w=copy.copy(node.taking_set_w))
                            node.add_child(child=child)
                            self.expand_node(node=child)
                elif node.T1:
                    t1_sets = copy.copy(node.T1)
                    #这里移除了填充的-1，不作扩展
                    if -1 in t1_sets:
                        t1_sets.remove(-1)
                    for t1_set in itertools.combinations(t1_sets, min(t3_num - len(node.AAA) - len(node.ABC),len(t1_sets))):
                        if t1_set:
                            T1 = copy.copy(node.T1)
                            raw = list(t1_set)
                            for t1 in raw:
                                T1.remove(t1)
                            child = Node_PH(AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                            T2=MJ.deepcopy(node.T2), T1=T1, jiang=copy.copy(node.jiang), raw=raw,
                                            taking_set=copy.copy(node.taking_set),
                                            taking_set_w=copy.copy(node.taking_set_w))
                            node.add_child(child=child)
                            self.expand_node(node=child)

        if self.cal_xts(node) == 2 and len(node.AAA) + len(node.ABC) == 3 and node.jiang and not node.raw:  # 定将完之后，只有3个刻子、顺子再加孤张牌的情况，用一张孤张牌扩展成T2使xts==1
            for t1 in node.T1:
                for item in t1tot2_dict[str(t1)]:  # "t1": [t2, t1_left, p]
                    T2 = MJ.deepcopy(node.T2)
                    T2.append(item[0])
                    T1 = copy.copy(node.T1)
                    T1.remove(t1)
                    taking_set = copy.copy(node.taking_set)
                    taking_set_w = copy.copy(node.taking_set_w)
                    taking_set.append(item[1])
                    taking_set_w.append(item[-1])
                    child = Node_PH(take=item[1], AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                    jiang=copy.copy(node.jiang), T2=T2, T1=T1,
                                    taking_set=taking_set, taking_set_w=taking_set_w)
                    node.add_child(child=child)
                    self.expand_node(node=child)

    def generate_tree(self):
        kz = []
        sz = []
        for t3 in self.suits:
            if t3[0] == t3[1]:
                kz.append([t3[0], t3[0], t3[0]])
            else:
                sz.append(t3)
        CS = self.combination_sets
        for cs in CS:
            root = Node_PH(take=None, AAA=cs[0] + kz, ABC=cs[1] + sz, jiang=[], T2=cs[2] + cs[3], T1=cs[-2],
                           taking_set=[], taking_set_w=[])
            self.tree_dict.append(root)
            self.expand_node(node=root)
            #traverse(root)

    def cards_type_evaluation(self, node: Node_PH):
        """
        功能：牌型评估值的计算，计算平胡搜索树中所有达到xts为1的节点的评估值，赋值给平胡牌型
        :param node:
        :return:
        """
        if node.xts == 1:
            self.type_evaluation += node.fan_value
        elif node.children:
            for child in node.children:
                self.cards_type_evaluation(node=child)

    def get_fan_score(self):
        self.generate_tree()
        for root in self.tree_dict:
            self.cards_type_evaluation(root)
        return self.type_evaluation

    def calculate_path_expectation(self, node):
        #深度搜索
        if node.ting_info:
            self.node_num += 1
            discard_set = []
            for t2 in node.T2:
                discard_set.extend(t2)
            discard_set.extend(node.T1)
            taking_set_sorted = sorted(node.taking_set)
            taking_set_lable = str(taking_set_sorted)  # 转化为str可以加快查找
            if discard_set == []:
                logger.info("pinghu_error: AAA:%s, ABC:%s, jiang:%s, T2:%s, T1:%s", node.AAA, node.ABC, node.jiang, node.T2, node.T1)
                return
            # todo 这种按摸牌的评估方式是否唯一准确
            for card in list(set(discard_set)):
                score = 0
                for info in node.ting_info:
                    if card not in info['cs']:
                        score += info['score']
                if card not in self.discard_state.keys():
                    self.discard_state[card] = [[], []]
                if taking_set_lable not in self.discard_state[card][0]:
                    self.discard_state[card][0].append(taking_set_lable)
                    self.discard_state[card][-1].append(score)
                else:
                    index = self.discard_state[card][0].index(taking_set_lable)
                    if score > self.discard_state[card][-1][index]:
                        self.chang_num += 1
                        self.discard_state[card][-1][index] = score

        elif node.children != []:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):
        for root in self.tree_dict:
            self.calculate_path_expectation(root)
        state_num = 0
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][-1])
            state_num += len(self.discard_state[discard][-1])
        return self.discard_score

class Node_Qidui:
    def __init__(self, take=None, AA=[], T1=[], raw=[], taking_set=[]):
        self.take = take
        self.AA = AA
        self.T1 = T1
        self.raw = raw
        self.xts = 14
        self.taking_set = taking_set
        self.children = []
        self.ting_info = []  # 手牌达到听牌时的听牌信息，包括听牌，剩余牌，对应的番数  [{ting_card: card, remain_num: num, fan:num}, ...]
        self.path_value = None
        self.fan_value = None

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print("AA:", self.AA, "T1:", self.T1, "raw:", self.raw, "taking_set:", self.taking_set, "\nting_info:",
              self.ting_info, "path_value:", self.path_value, "fan_value:", self.fan_value)

class Qidui:
    def __init__(self, cards, suits, padding=[]):
        self.cards = cards
        self.suits = suits
        self.discard_score = {}
        self.padding = padding
        self.tree_dict = []
        self.combination_sets = []
        self.discard_state = {}
        self.type_evaluation = 0
        self.node_num = 0

    def qidui_CS(self):
        CS = [[], [], 14]
        if self.suits:
            return CS
        for card in list(set(self.cards)):
            n = self.cards.count(card)
            if n == 1:
                CS[1].append(card)
            elif n == 2:
                CS[0].append([card, card])
            elif n == 3:
                CS[0].append([card, card])
                CS[1].append(card)
            elif n == 4:
                CS[0].append([card, card])
                CS[0].append([card, card])
        CS[-1] -= len(CS[0])*2 + (7-len(CS[0]))
        self.combination_sets = CS
        return CS

    def fan(self, AAs):
        """
        七对番型
        :param AAs:
        """
        fan = 4
        yaojiu = [1, 9, 0x11, 0x19, 0x21, 0x29]
        flag_yaojiu = True
        flag_duanyaojiu = True
        for AA in AAs:
            if AA[0] not in yaojiu:
                flag_yaojiu = False
                break
        if flag_yaojiu:
            fan *= 4
        else:
            for AA in AAs:
                if AA[0] in yaojiu:
                    flag_duanyaojiu = False
                    break
            if flag_duanyaojiu:
                fan *= 2
        # 清七对判断
        color = AAs[0][0] & 0xF0
        qingyise = True
        for AA in AAs:
            if AA[0] & 0xF0 != color:
                qingyise = False
                break
        if qingyise:
            fan *= 4
        # 龙七对的判断
        for AA in AAs:
            if AAs.count(AA) == 2:
                fan *= 2
        return fan

    def ting_module(self, node):
        """
        功能：计算手牌达到听牌时能听哪些牌，听牌数，所胡的番型
        :param node:
        :return: [ {ting_card: card, remain_num: num, fan: int}, {ting_card: card, remain_num: num, fan: int}, ...]
        """
        ting_info = []
        for t1 in node.T1:
            remain_num = LEFT_NUM[MJ.convert_hex2index(t1)]
            AAs = node.AA + [[t1, t1]]
            fan = self.fan(AAs=AAs)
            ting = {'ting_card': t1, 'remain_num': remain_num, 'fan': fan}
            ting_info.append(ting)
        node.ting_info = ting_info

    def cal_score(self, node):
        path_value = 1
        for i in range(len(node.taking_set)):
            card = node.taking_set[i]
            taking_rate = T_SELFMO[MJ.convert_hex2index(card)]
            path_value *= taking_rate

        # 摸牌概率修正，当一张牌被重复获取时，T_selfmo修改为当前数量占未出现牌数量的比例
        taking_set = list(set(node.taking_set))
        taking_set_num = [node.taking_set.count(i) for i in taking_set]
        for i in range(len(taking_set_num)):
            n = taking_set_num[i]
            j = 0
            while n > 1:
                j += 1
                index = MJ.convert_hex2index(taking_set[i])
                if LEFT_NUM[index] >= n:
                    path_value *= float(LEFT_NUM[index] - j) / LEFT_NUM[index]
                else:  # 摸牌数超过了剩余数，直接舍弃
                    return 0
                n -= 1
        if path_value:
            ting_evaluate = 0
            for info in node.ting_info:
                if info['remain_num']:
                    ting_evaluate += info['remain_num'] * info['fan']
            fan_value = path_value * ting_evaluate
            node.path_value = path_value
            node.fan_value = fan_value
        else:
            node.path_value = 0
            node.fan_value = 0


    def expand_node(self, node):
        """
        功能：七对搜索树节点的扩展，扩展至xts=6，即凑成6对对子即可
        :param node:
        :return:
        """
        if len(node.AA) == 6:
            node.xts = 1
            self.node_num += 1
            self.ting_module(node=node)
            self.cal_score(node=node)
            return
        else:
            if node.raw:
                card = node.raw[-1]
                node.raw.pop()
                AA = copy.copy(node.AA)
                AA.append([card, card])
                taking_set = copy.copy(node.taking_set)
                taking_set.append(card)
                child = Node_Qidui(take=card, AA=AA, T1=node.T1, raw=node.raw, taking_set=taking_set)
                node.add_child(child=child)
                self.expand_node(node=child)
            else:
                if node.T1:
                    t1_sets = copy.copy(node.T1)
                    if -1 in t1_sets:
                        t1_sets.remove(-1)
                    T1 = copy.copy(node.T1)
                    for t1_set in itertools.combinations(t1_sets, min(6 - len(node.AA), len(t1_sets))):
                        node.T1 = copy.copy(T1)
                        node.raw = list(t1_set)
                        for t1 in node.raw:
                            node.T1.remove(t1)
                        self.expand_node(node=node)

    def generate_tree(self):
        CS = self.combination_sets
        node = Node_Qidui(take=None, AA=CS[0], T1=CS[1], taking_set=[])
        self.tree_dict.append(node)
        self.expand_node(node=node)
        # traverse(node)

    def cards_type_evaluation(self, node):
        if node.xts == 1:
            self.type_evaluation += node.fan_value
        elif node.children:
            for child in node.children:
                self.cards_type_evaluation(node=child)

    def get_fan_score(self):
        self.generate_tree()
        for root in self.tree_dict:
            self.cards_type_evaluation(root)
        return self.type_evaluation

    def evaluate(self, node):
        if not node.children:
            if node.ting_info:
                taking_set_sorted = sorted(node.taking_set)
                value = 1
                for card in taking_set_sorted:
                    if card == -1:
                        value = 1.0/34
                    else:
                        value *= T_SELFMO[MJ.convert_hex2index(card)]
                discards = node.T1+self.padding
                for discard in discards:
                    score = 0
                    for info in node.ting_info:
                        if info['ting_card'] != discard:
                            score += value * info['fan'] * info['remain_num']
                    if discard not in self.discard_state.keys():
                        self.discard_state[discard] = [[], []]
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
                    elif taking_set_sorted not in self.discard_state[discard][0]:
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
        else:
            for child in node.children:
                self.evaluate(child)

    def get_discard_score(self):
        for tree in self.tree_dict:
            self.evaluate(tree)
        for discard in self.discard_state.keys():
            if discard not in self.discard_score:
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][-1])
        return self.discard_score

class Node_Yaojiu:
    def __init__(self, take=None, AAA=[], ABC=[], T2=[], T1=[], jiang=[], raw=[], taking_set=[], taking_set_w=[], useless_cards=[]):
        self.take = take
        self.AAA = AAA
        self.ABC = ABC
        self.T2 = T2
        self.T1 = T1
        self.jiang = jiang
        self.raw = raw
        self.xts = 14
        self.taking_set = taking_set
        self.taking_set_w = taking_set_w
        self.useless_cards = useless_cards
        self.children = []
        self.ting_info = []
        self.path_value = None
        self.fan_value = None

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print("AAA:", self.AAA, "ABC:", self.ABC, "jiang:", self.jiang, "T2:", self.T2, "T1:", self.T1,
              "taking_set:", self.taking_set, "raw:", self.raw, "\nuseless_cards:", self.useless_cards,
              "ting_info:", self.ting_info,"path_value:", self.path_value, "fan_value:", self.fan_value, "xts:", self.xts)

class Yaojiu:
    def __init__(self, cards, suits, padding=[]):
        self.cards = cards
        self.suits = suits
        self.discard_score = {}
        self.padding = padding
        self.discard_state = {}
        self.tree_dict = []
        self.combination_sets = []
        self.yaojiu = [1, 9, 0x11, 0x19, 0x21, 0x29]
        self.type_evaluation = 0
        self.node_num = 0

    def yaojiu_CS(self):
        """
        功能：计算幺九牌型的组合
        思路：4、5、6三张牌直接放入废牌中，对剩下的牌进行组合，对于不包含幺九牌的组合只能是单张牌或者是搭子，包含幺九牌的刻子、顺子直接放入组合中，
        :return: [[幺九刻子], [幺九顺子], [aa], [ab,ac], xts, [T1], [4、5、6废牌]]
        """
        PH = MJ.PingHu(cards=MJ.deepcopy(self.cards), suits=MJ.deepcopy(self.suits), padding=self.padding, fan_type=2, useless_cards=[])
        CS_YJ = PH.pinghu_CS()
        self.combination_sets = CS_YJ
        return CS_YJ

    def fan(self, node, ting_card, cs=[]):
        """
        功能：计算幺九手牌的番型
        :param node:
        :return:
        """
        fan = 4
        AAA = MJ.deepcopy(node.AAA)
        ABC = MJ.deepcopy(node.ABC)
        jiang = copy.copy(node.jiang)
        if jiang:
            t3 = cs + [ting_card]
            if t3[0] == t3[1]:
                AAA.append(t3)
            else:
                t3.sort()
                ABC.append(t3)
        else:
            jiang = [ting_card, ting_card]

        # qinyise
        flag_qinyise = True
        color = ting_card & 0xF0
        for t3 in AAA + ABC:
            if t3[0] & 0xF0 != color:
                flag_qinyise = False
                break
        if jiang[0] & 0xF0 != color:
            flag_qinyise = False
        if flag_qinyise:
            fan *= 4

        # pengpenghu
        if len(AAA) == 4:
            fan *= 2

        # jinggougou
        if len(self.suits) == 4:
            fan *= 2
        for suit in self.suits:
            if len(suit) == 4:
                fan *= 2
        return fan

    def ting_module(self, node):
        """
        功能：听牌模块，计算当前节点所听牌及其剩余牌数，番型
        :param node:
        :return:  [ {ting_card: card, remain_num: num, fan: int}, {ting_card: card, remain_num: num, fan: int}, ...]
        """
        ting_info = []
        if node.jiang:  # 将牌存在
            for t2 in node.T2:
                ting_cards = MJ.get_effective_cards(t2)
                for ting_card in ting_cards:
                    t3 = t2 + [ting_card]
                    t3.sort()
                    if t3[0] not in self.yaojiu and t3[2] not in self.yaojiu:
                        continue
                    remain_num = LEFT_NUM[MJ.convert_hex2index(ting_card)]
                    fan = self.fan(node, ting_card, t2)
                    ting = {'ting_card': ting_card, 'cs': t2, 'remain_num': remain_num, 'fan': fan}
                    ting_info.append(ting)
        else:
            ting_cards = copy.copy(node.T1)
            for t2 in node.T2:
                if t2[0] == t2[1]:
                    node.ting_info = ting_info
                    return
                ting_cards.extend(t2)
            for ting_card in ting_cards:
                if ting_card not in self.yaojiu:
                    continue
                remain_num = LEFT_NUM[MJ.convert_hex2index(ting_card)]
                fan = self.fan(node, ting_card)
                ting = {'ting_card': ting_card, 'cs': [ting_card], 'remain_num': remain_num, 'fan': fan}
                ting_info.append(ting)
        node.ting_info = ting_info

    def cal_xts(self, node: Node_Yaojiu):
        """
         功能：计算节点的向听数
        思路：初始向听数为14，减去相应已成型的组合（kz,sz为３，aa/ab为２），当２Ｎ过剩时，只减去还需要的２Ｎ，对２Ｎ不足时，对还缺少的３Ｎ减去１，表示从孤张牌中选择一张作为３Ｎ的待选
        :param all: [[]]组合信息
        :param suits: 副露
        :return: all　计算向听数后的组合信息
        """
        t3N = node.AAA + node.ABC
        xts = 14 - len(t3N) * 3
        if node.jiang:
            if len(t3N) + len(node.T2) >= 4:
                xts -= (4 - len(t3N)) * 2 + 2
            else:
                xts -= (len(node.T2)) * 2 + 2 + 4 - (len(t3N) + len(node.T2))
        else:
            jiangs = []
            if len(t3N) + len(node.T2) >= 4:
                xts -= (4 - len(t3N)) * 2
                for t2 in node.T2:
                    jiangs += t2
            else:
                xts -= len(node.T2) * 2 + 4 - (len(t3N) + len(node.T2))
            jiangs += node.T1
            for jiang in jiangs:   # 在未定将时，如果孤张牌中没有幺九牌，xts不能直接-1
                if jiang in self.yaojiu:
                    xts -= 1
                    break
        return xts

    def cal_score(self, node):
        path_value = cal_path_value(copy.copy(node.taking_set), copy.copy(node.taking_set_w))
        if path_value:
            ting_evaluate = 0
            for info in node.ting_info:
                if info['remain_num']:
                    ting_evaluate += info['remain_num'] * info['fan']
            fan_value = path_value * ting_evaluate
            node.path_value = path_value
            node.fan_value = fan_value
        else:
            node.path_value = 0
            node.fan_value = 0

    def expand_node(self, node):
        """
        功能：幺九牌搜索树节点扩展，
        扩展策略：定将、补刻子、顺子同时进行
        :param node:
        :return:
        """
        # 胡牌判断
        if not node.raw and self.cal_xts(node) == 1:
            node.xts = 1
            self.node_num += 1
            self.ting_module(node=node)
            self.cal_score(node=node)
            if node.ting_info:
                for info in node.ting_info:
                    info['score'] = node.path_value * info['remain_num'] * info['fan']
            return

        has_jiang = False
        if not node.jiang and not node.raw:   # 将牌的扩展，用对子或用孤张牌
            if len(node.ABC) + len(node.AAA) + len(node.T2) > 4:
                for t2 in node.T2:
                    if t2[0] == t2[1]:
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        child = Node_Yaojiu(take=-1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2,
                                            T1=copy.copy(node.T1), jiang=t2, raw=MJ.deepcopy(node.raw),
                                            taking_set=copy.copy(node.taking_set),
                                            taking_set_w=copy.copy(node.taking_set_w),
                                            useless_cards=copy.copy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                        has_jiang = True

            t2_jiang = False
            if len(node.ABC) + len(node.AAA) + len(node.T2) <= 4:
                for t2 in node.T2:
                    if t2[0] == t2[1]:
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        child = Node_Yaojiu(take=-1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2,
                                            T1=copy.copy(node.T1), jiang=t2, raw=MJ.deepcopy(node.raw),
                                            taking_set=copy.copy(node.taking_set),
                                            taking_set_w=copy.copy(node.taking_set_w),
                                            useless_cards=copy.copy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                        t2_jiang = True

            if not has_jiang and not t2_jiang:
                jiangs = copy.copy(node.T1)
                if not jiangs:
                    for t2 in node.T2:
                        jiangs = t2
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        for t1 in jiangs:
                            if t1 in self.yaojiu:
                                taking_set = copy.copy(node.taking_set)
                                taking_set.append(t1)
                                taking_set_w = copy.copy(node.taking_set_w)
                                taking_set_w.append(1)
                                T1 = copy.copy(jiangs)
                                T1.remove(t1)
                                child = Node_Yaojiu(take=t1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                                    jiang=[t1, t1], T2=T2, T1=T1, raw=MJ.deepcopy(node.raw),
                                                    taking_set=taking_set, taking_set_w=taking_set_w,
                                                    useless_cards=copy.copy(node.useless_cards))
                                node.add_child(child=child)
                                self.expand_node(node=child)
                else:
                    for t1 in jiangs:
                        if t1 == -1 or t1 not in self.yaojiu:  # 对-1不作扩展
                            continue
                        taking_set = copy.copy(node.taking_set)
                        taking_set.append(t1)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.append(1)
                        T1 = copy.copy(jiangs)
                        T1.remove(t1)
                        child = Node_Yaojiu(take=t1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), jiang=[t1, t1],
                                            T2=MJ.deepcopy(node.T2), T1=T1, raw=MJ.deepcopy(node.raw), taking_set=taking_set,
                                            taking_set_w=taking_set_w, useless_cards=copy.copy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)

        has_yaojiu = False
        for t1 in node.T1:
            if t1 in self.yaojiu:
                has_yaojiu = True
        for t2 in node.T2:
            if t2[0] in self.yaojiu or t2[1] in self.yaojiu:
                has_yaojiu = True
        if self.cal_xts(node) > 1 and not has_yaojiu and not node.jiang and not node.raw:  # 将牌为空，孤张牌和t2中没有幺九牌时，需要摸入一张幺九牌作为将牌扩展
            left_num = copy.copy(LEFT_NUM)
            for take in node.taking_set:
                left_num[MJ.convert_hex2index(take)] -= 1
            for yaojiu_card in self.yaojiu:
                if left_num[MJ.convert_hex2index(yaojiu_card)] > 1:
                    T1 = copy.copy(node.T1)
                    T1.append(yaojiu_card)
                    taking_set = copy.copy(node.taking_set)
                    taking_set_w = copy.copy(node.taking_set_w)
                    taking_set.append(yaojiu_card)
                    taking_set_w.append(1)
                    child = Node_Yaojiu(take=yaojiu_card, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                        jiang=copy.copy(node.jiang), T2=MJ.deepcopy(node.T2),
                                        T1=T1, raw=MJ.deepcopy(node.raw), taking_set=taking_set,
                                        taking_set_w=taking_set_w,
                                        useless_cards=copy.copy(node.useless_cards))
                    node.add_child(child=child)
                    self.expand_node(node=child)

        # T3扩展
        if len(node.AAA) + len(node.ABC) != 4 and not has_jiang:
            # 当待扩展集合不为空时，使用该集合进行扩展
            if node.raw:
                tn = node.raw[-1]
                raw = MJ.deepcopy(node.raw)  # 深度搜索后面的节点会改变raw，回退可能导致前面的节点raw不正确，这里需要copy
                raw.pop()
                if type(tn) == list:  # 使用t2扩展t3
                    t2 = tn
                    for item in t2tot3_dict[str(t2)]:  # "t2": [[t2_,t3,t1_left,valid,p]]
                        if item[1][0] not in self.yaojiu and item[1][2] not in self.yaojiu:
                            continue
                        AAA = MJ.deepcopy(node.AAA)
                        ABC = MJ.deepcopy(node.ABC)
                        if item[1][0] == item[1][1]:
                            AAA.append(item[1])
                        else:
                            ABC.append(item[1])
                        taking_set = copy.copy(node.taking_set)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set.append(item[-2])
                        taking_set_w.append(item[-1])
                        child = Node_Yaojiu(take=item[-2], AAA=AAA, ABC=ABC, jiang=copy.copy(node.jiang), T2=MJ.deepcopy(node.T2),
                                            T1=copy.copy(node.T1), raw=raw, taking_set=taking_set, taking_set_w=taking_set_w,
                                            useless_cards=copy.copy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                elif type(tn) == int:
                    t1 = tn
                    for item in t1tot3_dict[str(t1)]:  # {"t1":[[t3,t2(valid card),p]]}
                        if item[0][0] not in self.yaojiu and item[0][2] not in self.yaojiu:
                            continue
                        AAA = MJ.deepcopy(node.AAA)
                        ABC = MJ.deepcopy(node.ABC)
                        if item[0][0] == item[0][1]:
                            AAA.append(item[0])
                        else:
                            ABC.append(item[0])
                        taking_set = copy.copy(node.taking_set)
                        taking_set.extend(item[1])
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.extend(item[-1])
                        child = Node_Yaojiu(take=item[1], AAA=AAA, ABC=ABC, jiang=copy.copy(node.jiang),
                                            T2=MJ.deepcopy(node.T2), T1=copy.copy(node.T1),
                                            raw=raw, taking_set=taking_set, taking_set_w=taking_set_w,
                                            useless_cards=copy.copy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                else:
                    print("tn Error")
            else:
                t3_num = 3 if node.jiang else 4  # 如果已经定将，只需要凑齐3个刻子，顺子
                if node.T2:  # 1、先扩展T2为T3
                    t2_sets = MJ.deepcopy(node.T2)
                    for t2_set in itertools.combinations(t2_sets, min(t3_num - len(node.AAA) - len(node.ABC), len(t2_sets))):
                        if t2_set:
                            T2 = MJ.deepcopy(node.T2)
                            raw = list(t2_set)
                            for t2 in raw:
                                T2.remove(t2)
                            child = Node_Yaojiu(take=None, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2, T1=copy.copy(node.T1),
                                                jiang=copy.copy(node.jiang), raw=raw, taking_set=copy.copy(node.taking_set),
                                                taking_set_w=copy.copy(node.taking_set_w), useless_cards=copy.copy(node.useless_cards))
                            node.add_child(child=child)
                            self.expand_node(node=child)
                elif node.T1:
                    t1_sets = copy.copy(node.T1)
                    if -1 in t1_sets:  #这里移除了填充的-1，不作扩展
                        t1_sets.remove(-1)
                    for t1_set in itertools.combinations(t1_sets, min(t3_num - len(node.AAA) - len(node.ABC), len(t1_sets))):
                        if t1_set != ():
                            T1 = copy.copy(node.T1)
                            raw = list(t1_set)
                            for t1 in t1_set:
                                T1.remove(t1)
                            child = Node_Yaojiu(take=None, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=MJ.deepcopy(node.T2), T1=T1,
                                                jiang=copy.copy(node.jiang), raw=raw, taking_set=copy.copy(node.taking_set),
                                                taking_set_w=copy.copy(node.taking_set_w), useless_cards=copy.copy(node.useless_cards))
                            node.add_child(child=child)
                            self.expand_node(node=child)

        # 定将完之后，只有3个刻子、顺子再加孤张牌的情况，用一张孤张牌扩展成T2使xts==1
        if self.cal_xts(node) == 2 and len(node.AAA) + len(node.ABC) == 3 and node.jiang and not node.raw:
            for t1 in node.T1:
                for item in t1tot2_dict[str(t1)]:  # "t1": [t2, t1_left, p]
                    if item[0][0] not in self.yaojiu and item[0][1] not in self.yaojiu:
                        effect_card = MJ.get_effective_cards(item[0])
                        can_continue = False
                        for card in effect_card:
                            if card in self.yaojiu:
                                can_continue = True
                        if not can_continue:
                            continue
                    T2 = MJ.deepcopy(node.T2)
                    T2.append(item[0])
                    T1 = copy.copy(node.T1)
                    T1.remove(t1)
                    taking_set = copy.copy(node.taking_set)
                    taking_set_w = copy.copy(node.taking_set_w)
                    taking_set.append(item[1])
                    taking_set_w.append(item[-1])
                    child = Node_Yaojiu(take=item[1], AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                        jiang=copy.copy(node.jiang), T2=T2, T1=T1, raw=copy.copy(node.raw),
                                        taking_set=taking_set, taking_set_w=taking_set_w,
                                        useless_cards=copy.copy(node.useless_cards))
                    node.add_child(child=child)
                    self.expand_node(node=child)

        if self.cal_xts(node) > 1 and not node.T2 and not node.T1 and not node.raw:  # 手牌中的牌不够用时，需要再加一张牌
            taking_set = copy.copy(node.taking_set)
            taking_set_w = copy.copy(node.taking_set_w)
            left_num = copy.copy(LEFT_NUM)
            for take in taking_set:
                left_num[MJ.convert_hex2index(take)] -= 1
            for card in self.yaojiu:
                if left_num[MJ.convert_hex2index(card)]:
                    child = Node_Yaojiu(take=card, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), jiang=copy.copy(node.jiang), T1=[card],
                                        taking_set=taking_set + [card], taking_set_w=taking_set_w + [1],
                                        useless_cards=copy.copy(node.useless_cards))
                    node.add_child(child=child)
                    self.expand_node(node=child)

    def generate_tree(self):
        kz = []
        sz = []
        for t3 in self.suits:
            if t3[0] == t3[1]:
                kz.append([t3[0], t3[0], t3[0]])
            else:
                sz.append(t3)
        CS = self.combination_sets
        for cs in CS:
            root = Node_Yaojiu(take=None, AAA=cs[0]+kz, ABC=cs[1]+sz, T2=cs[2] + cs[3], T1=cs[-2], useless_cards=cs[-1])
            self.tree_dict.append(root)
            self.expand_node(node=root)
            #traverse(root)

    def cards_type_evaluation(self, node):
        if node.xts == 1:
            self.type_evaluation += node.fan_value
        else:
            for child in node.children:
                self.cards_type_evaluation(child)

    def get_fan_score(self):
        self.generate_tree()
        for root in self.tree_dict:
            self.cards_type_evaluation(root)
        return self.type_evaluation

    def calculate_path_expectation(self, node):
        # 深度搜索
        if node.ting_info and node.xts == 1:
            self.node_num += 1
            discard_set = []
            discard_set.extend(node.useless_cards)
            for t2 in node.T2:
                discard_set.extend(t2)
            discard_set.extend(node.T1)
            taking_set_sorted = sorted(node.taking_set)
            taking_set_lable = str(taking_set_sorted)  # 转化为str可以加快查找
            if discard_set == []:
                logger.info("yaojiu_error: AAA:%s, ABC:%s, jiang:%s, T2:%s, T1:%s", node.AAA, node.ABC, node.jiang,
                            node.T2, node.T1)
                return
            # todo 这种按摸牌的评估方式是否唯一准确
            for card in list(set(discard_set)):
                score = 0
                for info in node.ting_info:
                    if card not in info['cs']:
                        score += info['score']
                if card not in self.discard_state.keys():
                    self.discard_state[card] = [[], []]
                if taking_set_lable not in self.discard_state[card][0]:
                    self.discard_state[card][0].append(taking_set_lable)
                    self.discard_state[card][-1].append(score)
                else:
                    index = self.discard_state[card][0].index(taking_set_lable)
                    if score > self.discard_state[card][-1][index]:
                        self.discard_state[card][-1][index] = score

        elif node.children != []:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):
        for root in self.tree_dict:
            self.calculate_path_expectation(root)
        state_num = 0
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][-1])
            state_num += len(self.discard_state[discard][-1])
        # print("leaf node ", self.node_num)
        # print("state_num", state_num)
        # print("chang_num", self.chang_num)
        return self.discard_score

class Node_Duanyaojiu:
    def __init__(self, take=None, AAA=[], ABC=[], T2=[], T1=[], jiang=[], raw=[], taking_set=[], taking_set_w=[], useless_cards=[]):
        self.take = take
        self.AAA = AAA
        self.ABC = ABC
        self.T2 = T2
        self.T1 = T1
        self.jiang = jiang
        self.raw = raw
        self.xts = 14
        self.taking_set = taking_set
        self.taking_set_w = taking_set_w
        self.useless_cards = useless_cards
        self.children = []
        self.ting_info = []
        self.path_value = None
        self.fan_value = None

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print("AAA:", self.AAA, "ABC:", self.ABC, "jiang:", self.jiang, "T2:", self.T2, "T1:", self.T1,
              "taking_set:", self.taking_set, "taking_set_w:", self.taking_set_w,
              "raw:", self.raw, "\nuseless_cards:", self.useless_cards,
              "ting_info:", self.ting_info, "path_value:", self.path_value, "fan_value:", self.fan_value, "xts:",
              self.xts)

class Duanyaojiu:
    def __init__(self, cards=[], suits=[], padding=[]):
        self.cards = cards
        self.suits = suits
        self.discard_score = {}
        self.padding = padding
        self.discard_state = {}
        self.ting_info = []
        self.tree_dict = []
        self.combination_sets = []
        self.yaojiu = [1, 9, 0x11, 0x19, 0x21, 0x29]
        self.type_evaluation = 0
        self.chang_num = 0
        self.node_num = 0

    def duanyaojiu_CS(self):
        """
        功能：计算断幺九的所有组合
        思路：将手牌中的幺九牌去除之后再按照平胡的方法进行计算组合
        :return: [[刻子], [顺子], [aa], [ab], xts, [T1], [废牌]]
        """
        PH = MJ.PingHu(cards=copy.copy(self.cards), suits=MJ.deepcopy(self.suits), padding=self.padding, fan_type=3, useless_cards=[])
        CS_DYJ = PH.pinghu_CS()
        self.combination_sets = CS_DYJ
        return CS_DYJ

    def cal_xts(self, node: Node_Duanyaojiu):
        """
         功能：计算节点的向听数
        思路：初始向听数为14，减去相应已成型的组合（kz,sz为３，aa/ab为２），当２Ｎ过剩时，只减去还需要的２Ｎ，对２Ｎ不足时，对还缺少的３Ｎ减去１，表示从孤张牌中选择一张作为３Ｎ的待选
        :param all: [[]]组合信息
        :param suits: 副露
        :return: all　计算向听数后的组合信息
        """
        t3N = node.AAA + node.ABC
        xts = 14 - len(t3N) * 3
        has_jiang = False
        if node.jiang:
            has_jiang = True
        if has_jiang:
            if len(t3N) + len(node.T2) >= 4:
                xts -= (4 - len(t3N)) * 2 + 2
            else:
                xts -= (len(node.T2)) * 2 + 2 + 4 - (len(t3N) + len(node.T2))
        else:
            if len(t3N) + len(node.T2) >= 4:
                xts -= (4 - len(t3N)) * 2 + 1
            else:
                xts -= len(node.T2) * 2 + 1 + 4 - (len(t3N) + len(node.T2))
        return xts

    def fan(self, node, ting_card, cs=[]):
        """
        功能：计算断幺九手牌的番型
        :param node:
        :return:
        """
        fan = 2
        AAA = MJ.deepcopy(node.AAA)
        ABC = MJ.deepcopy(node.ABC)
        jiang = copy.copy(node.jiang)
        if jiang:
            t3 = cs + [ting_card]
            if t3[0] == t3[1]:
                AAA.append(t3)
            else:
                t3.sort()
                ABC.append(t3)
        else:
            jiang = [ting_card, ting_card]

        # qinyise
        flag_qinyise = True
        color = ting_card & 0xF0
        for t3 in AAA + ABC:
            if t3[0] & 0xF0 != color:
                flag_qinyise = False
                break
        if jiang[0] & 0xF0 != color:
            flag_qinyise = False
        if flag_qinyise:
            fan *= 4

        # pengpenghu
        if len(AAA) == 4:
            fan *= 2

        # jinggougou
        if len(self.suits) == 4:
            fan *= 2
        for suit in self.suits:
            if len(suit) == 4:
                fan *= 2
        return fan

    def ting_module(self, node):
        """
        功能：听牌模块，计算当前节点所听牌及其剩余牌数，番型
        :param node:
        :return:  [ {ting_card: card, remain_num: num, fan: int}, {ting_card: card, remain_num: num, fan: int}, ...]
        """
        ting_info = []
        if node.jiang:  # 将牌存在
            for t2 in node.T2:
                ting_cards = MJ.get_effective_cards(t2)
                for ting_card in ting_cards:
                    if ting_card in self.yaojiu:
                        continue
                    remain_num = LEFT_NUM[MJ.convert_hex2index(ting_card)]
                    fan = self.fan(node, ting_card, t2)
                    ting = {'ting_card': ting_card, 'cs': t2, 'remain_num': remain_num, 'fan': fan}
                    ting_info.append(ting)
        else:
            ting_cards = copy.copy(node.T1)
            for t2 in node.T2:
                if t2[0] == t2[1]:
                    node.ting_info = ting_info
                    return
                ting_cards += t2
            for ting_card in ting_cards:
                remain_num = LEFT_NUM[MJ.convert_hex2index(ting_card)]
                fan = self.fan(node, ting_card)
                ting = {'ting_card': ting_card, 'cs': [ting_card],'remain_num': remain_num, 'fan': fan}
                ting_info.append(ting)
        node.ting_info = ting_info

    def cal_score(self, node):
        path_value = cal_path_value(copy.copy(node.taking_set), copy.copy(node.taking_set_w))
        if path_value:
            ting_evaluate = 0
            for info in node.ting_info:
                if info['remain_num']:
                    ting_evaluate += info['remain_num'] * info['fan']
            fan_value = path_value * ting_evaluate
            node.path_value = path_value
            node.fan_value = fan_value
        else:
            node.path_value = 0
            node.fan_value = 0

    def expand_node(self, node):
        """
        功能：断幺九搜索树节点扩展，
        扩展策略：定将、补刻子、顺子同时进行
        :param node:
        :return:
        """
        # 胡牌判断
        if not node.raw and self.cal_xts(node) == 1:
            node.xts = 1
            self.node_num += 1
            self.ting_module(node=node)
            self.cal_score(node=node)
            if node.ting_info:
                for info in node.ting_info:
                    info['score'] = node.path_value * info['remain_num'] * info['fan']
            return

        has_jiang = False
        if not node.jiang and not node.raw:  # 将牌的扩展，用对子或用孤张牌
            if len(node.ABC) + len(node.AAA) + len(node.T2) > 4:
                for t2 in node.T2:
                    if t2[0] == t2[1]:
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        child = Node_Duanyaojiu(take=-1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2,
                                                T1=copy.copy(node.T1), jiang=t2, raw=MJ.deepcopy(node.raw),
                                                taking_set=copy.copy(node.taking_set),
                                                taking_set_w=copy.copy(node.taking_set_w),
                                                useless_cards=copy.copy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                        has_jiang = True
            t2_jiang = False
            if len(node.ABC) + len(node.AAA) + len(node.T2) <= 4:
                for t2 in node.T2:
                    if t2[0] == t2[1]:
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        child = Node_Duanyaojiu(take=-1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2,
                                                T1=copy.copy(node.T1), jiang=t2, raw=MJ.deepcopy(node.raw),
                                                taking_set=copy.copy(node.taking_set),
                                                taking_set_w=copy.copy(node.taking_set_w),
                                                useless_cards=copy.copy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                        t2_jiang = True

            if not has_jiang and not t2_jiang:
                jiangs = copy.copy(node.T1)
                if jiangs == []:
                    for t2 in node.T2:
                        jiangs = t2
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        for t1 in jiangs:
                            taking_set = copy.copy(node.taking_set)
                            taking_set.append(t1)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set_w.append(1)
                            T1 = copy.copy(jiangs)
                            T1.remove(t1)
                            child = Node_Duanyaojiu(take=t1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                                    jiang=[t1, t1], T2=T2, T1=T1, raw=MJ.deepcopy(node.raw),
                                                    taking_set=taking_set, taking_set_w=taking_set_w,
                                                    useless_cards=copy.copy(node.useless_cards))
                            node.add_child(child=child)
                            self.expand_node(node=child)
                else:
                    for t1 in jiangs:
                        if t1 == -1:  # 对-1不作扩展
                            continue
                        taking_set = copy.copy(node.taking_set)
                        taking_set.append(t1)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.append(1)
                        T1 = copy.copy(jiangs)
                        T1.remove(t1)
                        child = Node_Duanyaojiu(take=t1, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                                jiang=[t1, t1], T2=MJ.deepcopy(node.T2), T1=T1,
                                                raw=MJ.deepcopy(node.raw), taking_set=taking_set, taking_set_w=taking_set_w,
                                                useless_cards=copy.copy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)

        # T3扩展
        if len(node.AAA) + len(node.ABC) != 4 and not has_jiang:
            # 当待扩展集合不为空时，使用该集合进行扩展
            if node.raw:
                tn = node.raw[-1]
                raw = MJ.deepcopy(node.raw)  # 深度搜索后面的节点会改变raw，回退可能导致前面的节点raw不正确，这里需要copy
                raw.pop()
                if type(tn) == list:  # 使用t2扩展t3
                    t2 = tn
                    for item in t2tot3_dict[str(t2)]:  # "t2": [[t2_,t3,t1_left,valid,p]]
                        if item[1][0] in self.yaojiu or item[1][2] in self.yaojiu:
                            continue
                        if item[1][0] == item[1][1]:
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(item[1])
                            ABC = MJ.deepcopy(node.ABC)
                        else:
                            AAA = MJ.deepcopy(node.AAA)
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[1])
                        taking_set = copy.copy(node.taking_set)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set.append(item[-2])
                        taking_set_w.append(item[-1])
                        child = Node_Duanyaojiu(take=item[-2], AAA=AAA, ABC=ABC, jiang=MJ.deepcopy(node.jiang),
                                                T2=MJ.deepcopy(node.T2), T1=MJ.deepcopy(node.T1), raw=raw, taking_set=taking_set,
                                                taking_set_w=taking_set_w, useless_cards=MJ.deepcopy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                elif type(tn) == int:
                    t1 = tn
                    for item in t1tot3_dict[str(t1)]:  # {"t1":[[t3,t2(valid card),p]]}
                        if item[0][0] in self.yaojiu or item[0][2] in self.yaojiu:
                            continue
                        if item[0][0] == item[0][1]:
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(item[0])
                            ABC = MJ.deepcopy(node.ABC)
                        else:
                            AAA = MJ.deepcopy(node.AAA)
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[0])
                        take = item[1]
                        take_w = item[-1]
                        taking_set = copy.copy(node.taking_set)
                        taking_set.extend(take)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.extend(take_w)
                        child = Node_Duanyaojiu(take=take, AAA=AAA, ABC=ABC, jiang=MJ.deepcopy(node.jiang),
                                                T2=MJ.deepcopy(node.T2), T1=MJ.deepcopy(node.T1),
                                                raw=raw, taking_set=taking_set, taking_set_w=taking_set_w,
                                                useless_cards=MJ.deepcopy(node.useless_cards))
                        node.add_child(child=child)
                        self.expand_node(node=child)
                else:
                    print("tn Error")
            else:
                t3_num = 3 if node.jiang else 4  # 如果已经定将，只需要凑齐3个刻子，顺子
                if node.T2:  # 1、先扩展T2为T3
                    t2_sets = MJ.deepcopy(node.T2)
                    for t2_set in itertools.combinations(t2_sets, min(t3_num - len(node.AAA) - len(node.ABC), len(t2_sets))):
                        if t2_set:
                            T2 = copy.copy(node.T2)
                            raw = list(t2_set)
                            for t2 in raw:
                                T2.remove(t2)
                            child = Node_Duanyaojiu(take=None, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC), T2=T2,
                                                    T1=MJ.deepcopy(node.T1), jiang=MJ.deepcopy(node.jiang), raw=raw,
                                                    taking_set=MJ.deepcopy(node.taking_set),
                                                    taking_set_w=MJ.deepcopy(node.taking_set_w), useless_cards=MJ.deepcopy(node.useless_cards))
                            node.add_child(child=child)
                            self.expand_node(node=child)
                elif node.T1:
                    t1_sets = copy.copy(node.T1)
                    #这里移除了填充的-1，不作扩展
                    if -1 in t1_sets:
                        t1_sets.remove(-1)
                    for t1_set in itertools.combinations(t1_sets, min(t3_num - len(node.AAA) - len(node.ABC), len(t1_sets))):
                        if t1_set:
                            T1 = copy.copy(node.T1)
                            raw = list(t1_set)
                            for t1 in t1_set:
                                T1.remove(t1)
                            child = Node_Duanyaojiu(take=None, AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                                    T2=MJ.deepcopy(node.T2), T1=T1, jiang=MJ.deepcopy(node.jiang), raw=raw,
                                                    taking_set=MJ.deepcopy(node.taking_set), taking_set_w=MJ.deepcopy(node.taking_set_w),
                                                    useless_cards=MJ.deepcopy(node.useless_cards))
                            node.add_child(child=child)
                            self.expand_node(node=child)

        # 定将完之后，只有3个刻子、顺子再加孤张牌的情况，用一张孤张牌扩展成T2使xts==1
        if self.cal_xts(node) == 2 and len(node.AAA) + len(node.ABC) == 3 and node.jiang and not node.raw and not node.T2:
            for t1 in node.T1:
                for item in t1tot2_dict[str(t1)]:  # "t1": [t2, t1_left, p]
                    if item[1] in self.yaojiu:
                        continue
                    T2 = copy.copy(node.T2)
                    T2.append(item[0])
                    T1 = copy.copy(node.T1)
                    T1.remove(t1)
                    taking_set = copy.copy(node.taking_set)
                    taking_set_w = copy.copy(node.taking_set_w)
                    taking_set.append(item[1])
                    taking_set_w.append(item[-1])
                    child = Node_Duanyaojiu(take=item[1], AAA=MJ.deepcopy(node.AAA), ABC=MJ.deepcopy(node.ABC),
                                            jiang=MJ.deepcopy(node.jiang), T2=T2, T1=T1, raw=MJ.deepcopy(node.raw),
                                            taking_set=taking_set, taking_set_w=taking_set_w,
                                            useless_cards=MJ.deepcopy(node.useless_cards))
                    node.add_child(child=child)
                    self.expand_node(node=child)

    def generate_tree(self):
        kz = []
        sz = []
        for t3 in self.suits:
            if t3[0] == t3[1]:
                kz.append([t3[0], t3[0], t3[0]])
            else:
                sz.append(t3)
        CS = self.combination_sets
        for cs in CS:
            tree = Node_Duanyaojiu(take=None, AAA=cs[0]+kz, ABC=cs[1]+sz, T2=cs[2] + cs[3], T1=cs[-2], useless_cards=cs[-1])
            self.tree_dict.append(tree)
            self.expand_node(tree)
            #traverse(tree)

    def cards_type_evaluation(self, node):
        if node.xts == 1:
            self.type_evaluation += node.fan_value
        elif node.children:
            for child in node.children:
                self.cards_type_evaluation(node=child)

    def get_fan_score(self):
        """
        功能：计算断幺九牌型的牌型评估值
        思路：生成搜索树之后，对搜索树进行评估，计算断幺九牌的牌型评估值
        :return:
        """
        self.generate_tree()
        for root in self.tree_dict:
            self.cards_type_evaluation(root)
        return self.type_evaluation

    def calculate_path_expectation(self, node):
        #深度搜索
        if node.ting_info and node.xts == 1:
            self.node_num += 1
            discard_set = []
            discard_set.extend(node.useless_cards)
            for t2 in node.T2:
                discard_set.extend(t2)
            discard_set.extend(node.T1)
            taking_set_sorted = sorted(node.taking_set)
            taking_set_lable = str(taking_set_sorted)  # 转化为str可以加快查找
            if discard_set == []:
                logger.info("duanyaojiu_error: AAA:%s, ABC:%s, jiang:%s, T2:%s, T1:%s", node.AAA, node.ABC, node.jiang,
                            node.T2, node.T1)
                return
            # todo 这种按摸牌的评估方式是否唯一准确
            for card in list(set(discard_set)):
                score = 0
                for info in node.ting_info:
                    if card not in info['cs']:
                        score += info['score']
                if card not in self.discard_state.keys():
                    self.discard_state[card] = [[], []]
                if taking_set_lable not in self.discard_state[card][0]:
                    self.discard_state[card][0].append(taking_set_lable)
                    self.discard_state[card][-1].append(score)
                else:
                    index = self.discard_state[card][0].index(taking_set_lable)
                    if score > self.discard_state[card][-1][index]:
                        self.chang_num += 1
                        self.discard_state[card][-1][index] = score

        elif node.children != []:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):
        for root in self.tree_dict:
            self.calculate_path_expectation(root)
        state_num = 0
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][-1])
            state_num += len(self.discard_state[discard][-1])
        # print("leaf node ", self.node_num)
        # print("state_num", state_num)
        # print("chang_num", self.chang_num)
        return self.discard_score


def traverse(root):
    if root.children:
        for child in root.children:
            traverse(child)
    else:
        root.node_info()


def cal_path_value(taking_set, taking_set_w):
    value = 1
    if taking_set_w != []:
        for i in range(len(taking_set)):
            card = taking_set[i]
            taking_rate = T_SELFMO[MJ.convert_hex2index(card)]
            value *= taking_rate * taking_set_w[i]

    # 摸牌概率修正，当一张牌被重复获取时，T_selfmo修改为当前数量占未出现牌数量的比例
    taking_set_copy = list(set(taking_set))
    taking_set_num = [taking_set.count(i) for i in taking_set_copy]
    for i in range(len(taking_set_num)):
        n = taking_set_num[i]
        j = 0
        while n > 1:
            j += 1
            index = MJ.convert_hex2index(taking_set_copy[i])
            if LEFT_NUM[index] >= n:
                value *= float(LEFT_NUM[index] - j) / LEFT_NUM[index]
            else:  # 摸牌数超过了剩余数，直接舍弃
                return 0
            n -= 1
    return value

def trandfer_discards(discards, discards_op, handcards, hu_cards):
    """
    获取场面剩余牌数量
    计算手牌和场面牌的数量，再计算未知牌的数量
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param handcards: 手牌
    :param hu_cards: 已胡牌
    :return: left_num, discards_list　剩余牌列表，已出现的牌数量列表
    """
    discards_map = {0x01: 0, 0x02: 1, 0x03: 2, 0x04: 3, 0x05: 4, 0x06: 5, 0x07: 6, 0x08: 7, 0x09: 8,
                    0x11: 9, 0x12: 10, 0x13: 11, 0x14: 12, 0x15: 13, 0x16: 14, 0x17: 15, 0x18: 16, 0x19: 17,
                    0x21: 18, 0x22: 19, 0x23: 20, 0x24: 21, 0x25: 22, 0x26: 23, 0x27: 24, 0x28: 25, 0x29: 26}
    left_num = [4] * 27
    discards_list = [0] * 27
    for per in discards:
        for item in per:
            discards_list[discards_map[item]] += 1
            left_num[discards_map[item]] -= 1
    for seat_op in discards_op:
        for op in seat_op:
            for item in op:
                discards_list[discards_map[item]] += 1
                left_num[discards_map[item]] -= 1
    for item in handcards:
        left_num[discards_map[item]] -= 1
    for per in hu_cards:
        for card in per:
            discards_list[discards_map[card]] += 1
            left_num[discards_map[card]] -= 1
    left_num_print = {}
    index = 0
    # for i in range(3):
    #     j = 0
    #     for j in range(1,10):
    #         card = i * 0x10 + j
    #         key = str(card)
    #         left_num_print[key] = left_num[index]
    #         index += 1
    # print(left_num_print)
    return left_num, discards_list


def value_t1(card):
    value = 0
    if card!=-1:
        for e in t1tot3_dict[str(card)]:
            v = 1
            for i in range(len(e[1])):
                v *= T_SELFMO[MJ.convert_hex2index(e[1][i])] * e[-1][i]
            value += v
    return value

def recommend_c_type_card(cards=[], c_type=None):
    '''
    功能：若手牌中存在定缺牌，按定缺牌的优先级选择出牌
    :param cards: 手牌
    :param c_type: 定缺花色
    '''
    t1_type = [[], [], []]  # 按优先级分类
    for card in cards:
        if card & 0xF0 == c_type:
            if card & 0x0F in [1, 9]:
                t1_type[0].append(card)
            if card & 0x0F in [2, 8]:
                t1_type[1].append(card)
            else:
                t1_type[2].append(card)
    if t1_type[0]:
        return t1_type[0][0]
    elif t1_type[1]:
        return t1_type[1][0]
    else:
        return t1_type[2][0]

def get_score_dict(cards, suits, padding=[], max_xts=14):
    #寻找向听数在阈值内的牌型
    PH = SearchTree_PH(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    QYS = Qingyise(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    PPH = Pengpenghu(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    YJ = Yaojiu(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    DYJ = Duanyaojiu(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    QD = Qidui(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    #组合信息
    # start = time.time()
    CS_PH = PH.pinghu_CS()
    # print("CS_PH:--------------------------------")
    # for cs in CS_PH:
    #     print(cs)
    CS_QYS = QYS.qingyise_CS()
    # print("CS_QYS:-------------------------------")
    # for cs in CS_QYS:
    #     print(cs)
    CS_PPH = PPH.pengpenghu_CS()
    # print("CS_PPH:-------------------------------")
    # print(CS_PPH)
    CS_YJ = YJ.yaojiu_CS()
    # print("CS_YJ:-------------------------------")
    # for cs in CS_YJ:
    #     print(cs)
    CS_DYJ = DYJ.duanyaojiu_CS()
    # print("CS_DYJ:------------------------------")
    # for cs in CS_DYJ:
    #     print(cs)
    CS_QD = QD.qidui_CS()
    # print("CS_QD:------------------------------")
    # print(CS_QD)
    # print("cs_cost:", time.time() - start)

    #向听数
    xts_list = [CS_PH[0][-3], CS_QYS[0][-3], CS_PPH[-1], CS_YJ[0][-3], CS_DYJ[0][-3], CS_QD[-1]]
    # print("xts_list PH,QYS,PPH,YJ,DYJ,QD", xts_list)
    logger.info("xts PH,QYS,PPH,YJ,DYJ,QD:%s", xts_list)
    min_xts = min(xts_list[1:])  # 除平胡外的其他牌型的最小向听数
    if min_xts > max_xts+1:  # op中吃碰后向听数增加的情况，特别是打非平胡的牌型
        return {cards[-1]: 0}, min_xts
    type_list = []  # 需搜索的牌型
    for i in range(1, 6):
        if xts_list[i]-1 <= min_xts:
            type_list.append(i)
    if REMAIN_NUM > 20:
        if min_xts - xts_list[0] >= 3:
            type_list = [0]
        elif min_xts - xts_list[0] >= 2:
            type_list.append(0)
    else:
        if min_xts - xts_list[0] > 3:
            type_list = [0]
        elif min_xts - xts_list[0] > 2:
            type_list.append(0)
    # print("type_list:", type_list)
    fan_score_list = [0.0] * 6
    for i in type_list:
        if i == 0:
            fan_score_list[0] = PH.get_fan_score()
        elif i == 1:
            fan_score_list[1] = QYS.get_fan_score()
        elif i == 2:
            fan_score_list[2] = PPH.get_fan_score()
        elif i == 3:
            fan_score_list[3] = YJ.get_fan_score()
        elif i == 4:
            fan_score_list[4] = DYJ.get_fan_score()
        elif i == 5:
            fan_score_list[5] = QD.get_fan_score()
    if fan_score_list == [0.0] * 6 and 0 not in type_list:
        fan_score_list[0] = PH.get_fan_score()
    # print("fan_score_list:", fan_score_list)
    score_dict = {}
    max_fan = fan_score_list.index(max(fan_score_list))
    if max_fan == 0:
        score_dict = PH.get_discard_score()
    elif max_fan == 1:
        score_dict = QYS.get_discard_score()
    elif max_fan == 2:
        score_dict = PPH.get_discard_score()
    elif max_fan == 3:
        score_dict = YJ.get_discard_score()
    elif max_fan == 4:
        score_dict = DYJ.get_discard_score()
    elif max_fan == 5:
        score_dict = QD.get_discard_score()
    keys = []
    for key in score_dict.keys():
        if score_dict[key] == 0:
            keys.append(key)
    for key in keys:
        score_dict.pop(key)
    print("score_dict:", score_dict)
    return score_dict, min_xts

def get_score_dict_op(cards, suits, padding=[], max_xts=14):
    # 寻找向听数在阈值内的牌型
    print(cards,"+",suits)
    PH = SearchTree_PH(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    QYS = Qingyise(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    PPH = Pengpenghu(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    YJ = Yaojiu(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    DYJ = Duanyaojiu(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    QD = Qidui(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    # 组合信息
    CS_PH = PH.pinghu_CS()
    CS_QYS = QYS.qingyise_CS()
    CS_PPH = PPH.pengpenghu_CS()
    CS_YJ = YJ.yaojiu_CS()
    CS_DYJ = DYJ.duanyaojiu_CS()
    CS_QD = QD.qidui_CS()


    # 向听数 无 pph
    xts_list = [CS_PH[0][-3], CS_QYS[0][-3], CS_PPH[-1], CS_YJ[0][-3], CS_DYJ[0][-3], CS_QD[-1]+2]
    print("xts_list PH,QYS,PPH,YJ,DYJ,QD", xts_list)
    logger.info("xts PH,QYS,PPH,YJ,DYJ,QD:%s", xts_list)
    min_xts = min(xts_list[1:])  # 除平胡外的其他牌型的最小向听数

    type_list = []  # 需搜索的牌型
    if min_xts <= 4:
        for i in range(1, 6):
            if xts_list[i] - 1 <= min_xts:
                if xts_list[i] > 4 and i == 1:# 清一色超过4张会超时
                    continue
                type_list.append(i)
    if REMAIN_NUM > 30:
        # if min_xts - xts_list[0] >= 3:
        #     type_list = [0]
        if min_xts - xts_list[0] >= 2:
            type_list.append(0)
    else:
        if min_xts - xts_list[0] >= 2:
            type_list = [0]

    # print("type_list:", type_list)
    if len(type_list) > 3 and min_xts > 4:
        type_list = [0]
    fan_score_list = [0.0] * 6
    for i in type_list:
        if i == 0:
            fan_score_list[0] = PH.get_fan_score()
        elif i == 1:
            fan_score_list[1] = QYS.get_fan_score()
        elif i == 2:
            fan_score_list[2] = PPH.get_fan_score()
        elif i == 3:
            fan_score_list[3] = YJ.get_fan_score()
        elif i == 4:
            fan_score_list[4] = DYJ.get_fan_score()
        elif i == 5:
            fan_score_list[5] = QD.get_fan_score()
    # print("max_fan_score_list:", max(fan_score_list),fan_score_list.index(max(fan_score_list)))   # 该番型 fan * num * P(path)
    if fan_score_list == [0.0] * 6:
        fan_score_list[0] = PH.get_fan_score()
        # print("fan_score_list_ph:", fan_score_list)

    score_dict = {}

    max_fan = fan_score_list.index(max(fan_score_list))
    if max_fan == 0:
        score_dict = PH.get_discard_score()
    elif max_fan == 1:
        score_dict = QYS.get_discard_score()
    elif max_fan == 2:
        score_dict = PPH.get_discard_score()
    elif max_fan == 3:
        score_dict = YJ.get_discard_score()
    elif max_fan == 4:
        score_dict = DYJ.get_discard_score()
    elif max_fan == 5:
        score_dict = QD.get_discard_score()
    keys = []
    for key in score_dict.keys():
        if score_dict[key] == 0:
            keys.append(key)
    for key in keys:
        score_dict.pop(key)
    # print("score_dict:", score_dict)
    # print("-------------------------------------------------------------------")
    return score_dict, min_xts


def get_score_dict_op_bak(cards, suits, padding=[], max_xts=14):
    #寻找向听数在阈值内的牌型
    PH = SearchTree_PH(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    QYS = Qingyise(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    PPH = Pengpenghu(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    YJ = Yaojiu(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    DYJ = Duanyaojiu(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    QD = Qidui(cards=copy.copy(cards), suits=copy.copy(suits), padding=padding)
    #组合信息
    # start = time.time()
    CS_PH = PH.pinghu_CS()
    # print("CS_PH:--------------------------------")
    # for cs in CS_PH:
    #     print(cs)
    CS_QYS = QYS.qingyise_CS()
    # print("CS_QYS:-------------------------------")
    # for cs in CS_QYS:
    #     print(cs)
    CS_PPH = PPH.pengpenghu_CS()
    # print("CS_PPH:-------------------------------")
    # print(CS_PPH)
    CS_YJ = YJ.yaojiu_CS()
    # print("CS_YJ:-------------------------------")
    # for cs in CS_YJ:
    #     print(cs)
    CS_DYJ = DYJ.duanyaojiu_CS()
    # print("CS_DYJ:------------------------------")
    # for cs in CS_DYJ:
    #     print(cs)
    CS_QD = QD.qidui_CS()
    # print("CS_QD:------------------------------")
    # print(CS_QD)
    # print("cs_cost:", time.time() - start)

    #向听数
    xts_list = [CS_PH[0][-3], CS_QYS[0][-3], CS_PPH[-1], CS_YJ[0][-3], CS_DYJ[0][-3], CS_QD[-1]+2]
    print("xts_list PH,QYS,PPH,YJ,DYJ,QD", xts_list)
    logger.info("xts PH,QYS,PPH,YJ,DYJ,QD:%s", xts_list)
    min_xts = min(xts_list[1:])  # 除平胡外的其他牌型的最小向听数

    type_list = []  # 需搜索的牌型
    if min_xts <= 4:
        for i in range(1, 6):
            if xts_list[i] - 1 <= min_xts:
                if i == 1 and xts_list[i] > 4: # 清一色超过4张会超时
                    continue
                type_list.append(i)
    if xts_list[0] != 0:  # 平胡的向听数为0时说明是在弃胡操作之后进行的出牌决策
        if REMAIN_NUM > 30:
            if min_xts - xts_list[0] >= 2:
                type_list.append(0)
        else:
            if min_xts - xts_list[0] >= 2:
                type_list = [0]
        if len(type_list) > 3 and min_xts > 4:
            type_list = [0]
    # print("type_list:", type_list)
    remain_fans = []  # 存放为进行牌型扩展的番型及其xts
    if xts_list[0] != 0 and 0 not in type_list:
        remain_fans.append({'fan': 0, 'xts': xts_list[0]})
    for i in range(1, 6):
        if i not in type_list and xts_list[i] != 14:
            remain_fans.append({'fan': i, 'xts': xts_list[i]})
    fan_score_list = [0.0] * 6
    for i in type_list:
        if i == 0:
            fan_score_list[0] = PH.get_fan_score()
        elif i == 1:
            fan_score_list[1] = QYS.get_fan_score()
        elif i == 2:
            fan_score_list[2] = PPH.get_fan_score()
        elif i == 3:
            fan_score_list[3] = YJ.get_fan_score()
        elif i == 4:
            fan_score_list[4] = DYJ.get_fan_score()
        elif i == 5:
            fan_score_list[5] = QD.get_fan_score()
    if fan_score_list == [0.0] * 6 and 0 not in type_list and xts_list[0] != 0:
        fan_score_list[0] = PH.get_fan_score()
        remain_fans.pop(0)
    score_dict = {}
    while not score_dict and fan_score_list != [0.0] * 6:
        max_fan = fan_score_list.index(max(fan_score_list))
        fan_score_list[max_fan] = 0
        if max_fan == 0:
            score_dict = PH.get_discard_score()
        elif max_fan == 1:
            score_dict = QYS.get_discard_score()
        elif max_fan == 2:
            score_dict = PPH.get_discard_score()
        elif max_fan == 3:
            score_dict = YJ.get_discard_score()
        elif max_fan == 4:
            score_dict = DYJ.get_discard_score()
        elif max_fan == 5:
            score_dict = QD.get_discard_score()
        keys = []
        for key in score_dict.keys():
            if score_dict[key] == 0:
                keys.append(key)
        for key in keys:
            score_dict.pop(key)
    remain_fans.sort(key=lambda fan: fan['xts'])
    while not score_dict and remain_fans:  # 如果计算完之后，score_dict为空说明没有正常出牌，需要计算其他番型
        max_fan = remain_fans[0]['fan']
        remain_fans.pop(0)
        if max_fan == 0:
            PH.get_fan_score()
            score_dict = PH.get_discard_score()
        elif max_fan == 1:
            QYS.get_fan_score()
            score_dict = QYS.get_discard_score()
        elif max_fan == 2:
            PPH.get_fan_score()
            score_dict = PPH.get_discard_score()
        elif max_fan == 3:
            YJ.get_fan_score()
            score_dict = YJ.get_discard_score()
        elif max_fan == 4:
            DYJ.get_fan_score()
            score_dict = DYJ.get_discard_score()
        elif max_fan == 5:
            QD.get_fan_score()
            score_dict = QD.get_discard_score()
        keys = []
        for key in score_dict.keys():
            if score_dict[key] == 0:
                keys.append(key)
        for key in keys:
            score_dict.pop(key)
    if not score_dict:  # 如果到最后score_dict还为空，则从xts数最小的番型中找到T1作为出牌，从手牌的表面看起来出牌正常
        min_xts_index = xts_list.index(min(xts_list))
        t1 = []
        if min_xts_index == 0:
            t1 = CS_PH[0][-2]
        elif min_xts_index == 1:
            t1 = CS_QYS[0][-1] + CS_QYS[0][-2]
        elif min_xts_index == 2:
            t1 = CS_PPH[0][-2]
        elif min_xts_index == 3:
            t1 = CS_YJ[0][-1] + CS_YJ[0][-2]
        elif min_xts_index == 4:
            t1 = CS_DYJ[0][-1] + CS_DYJ[0][-2]
        elif min_xts_index == 5:
            t1 = CS_QD[-2]
        for t in t1:
            score_dict[t] = 0.000001
    print("----------------------------------------------------")
    print('score_dict:', score_dict)
    return score_dict, min_xts


def recommend_switch_cards(hand_cards=[], switch_n_cards=3):
    switch_cards = SwitchTiles(hand=hand_cards, n=switch_n_cards).choose_3card()
    return switch_cards

def recommend_choose_color(hand_cards=[], switch_n_cards=3):
    choose_color = SwitchTiles(hand=hand_cards, n=switch_n_cards).choose_color_final()
    return choose_color

def recommend_card(cards=[], suits=[], discards=[], discards_op=[], remain_num=136, round=0, hu_cards=[], seat_id=0, c_type=0x40):
    """
    功能：推荐出牌接口
    思路：先用手牌对每种牌型做一个牌型匹配，匹配度到高的牌型进行一个评估，评估值高的作为胡牌的番型
    :param cards: 手牌
    :param suits: 副露
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param remain_num: 剩余牌
    :param round: 轮数
    :param seat_id: 座位号
    :param c_type: 定缺花色
    :return: outCard 推荐出牌
    """
    # 定缺牌的出牌
    has_c_type = False
    for card in cards:
        if card & 0xF0 == c_type:
            has_c_type = True
            break
    if has_c_type:
        return recommend_c_type_card(cards=cards, c_type=c_type)
    # 更新全局变量
    global T_SELFMO, LEFT_NUM, t2tot3_dict, t1tot3_dict, t1tot2_dict, TIME_START, RT1, RT2, RT3, ROUND
    ROUND = round
    TIME_START = time.time()
    LEFT_NUM, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards, hu_cards=hu_cards)
    REMAIN_NUM = max(1, min(sum(LEFT_NUM), remain_num))

    if True:
        T_SELFMO = [float(i) / REMAIN_NUM for i in LEFT_NUM]
        RT1 = []
        RT2 = []
        RT3 = []
    t1tot2_dict = MJ.t1tot2_info()
    t1tot3_dict = MJ.t1tot3_info()
    t2tot3_dict = MJ.t2tot3_info()

    score_dict, _ = get_score_dict(cards, suits)
    if score_dict:
        recommend_card = max(score_dict, key=lambda x: score_dict[x])
    else:
        recommend_card = cards[-1]
        logger.error("no card be recommonded,cards=%s,suits=%s",cards, suits)
    end = time.time()
    if end - TIME_START > 10:
        logger.error("overtime %s,%s,%s,%s", end - TIME_START, cards, suits)
    logger.info("recommend_card %s",recommend_card)
    return recommend_card

def recommend_op(op_card, cards=[], suits=[], discards=[], discards_op=[], c_type = 0x40 ,win_player_num=0,canchi=False,
                 self_turn=False, isHu=False, remain_num=10,hu_cards = [[],[],[],[]]):
    """
    功能：动作决策接口
    思路：使用向听数作为牌型选择依据，对最小ｘｔｓ的牌型，再调用相应的牌型类动作决策
    :param op_card: 操作牌
    :param cards: 手牌re
    :param suits: 副露
    :param king_card: 宝牌
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param canchi: 吃牌权限
    :param self_turn: 是否是自己回合
    :param fei_king: 飞宝数
    :param isHu: 是否胡牌
    :return: [],isHu 动作组合牌，是否胡牌
    """
    if isHu:
        try:
            res = choose_hu(handcards=cards, suits=suits, discards=discards, discards_op=discards_op,
                            win_player_num=win_player_num, remain_num=remain_num, op_card=op_card, hu_cards = hu_cards,zi_mo=False)
            return [], res
        except:
            return [],True

    # 更新全局变量
    global T_SELFMO, LEFT_NUM, t2tot3_dict, t1tot3_dict, TIME_START
    TIME_START = time.time()
    LEFT_NUM, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards,hu_cards=hu_cards)
    if len(cards) % 3 == 2:
        self_turn = True
    REMAIN_NUM = sum(LEFT_NUM)
    if remain_num // 4 > 100:
        T_SELFMO = []
        RT1 = []
        RT2 = []
        RT3 = []
    else:
        T_SELFMO = [float(i) / REMAIN_NUM for i in LEFT_NUM]
        RT1 = []
        RT2 = []
        RT3 = []

    t1tot3_dict = MJ.t1tot3_info()
    t2tot3_dict = MJ.t2tot3_info()
    # 去除定缺牌
    c_cards = []
    for card in cards:
        if card & 0xF0 == c_type:
            c_cards.append(card)

    for card in c_cards:
        cards.remove((card))
    set_cards = list(set(cards))

    if self_turn:  # 自己回合，暗杠或补杠
        for card in set_cards:
            if cards.count(card) == 4:
                return [card, card, card, card], False  # 暗杠必杠
        for suit in suits:
            if suit.count(suit[0]) == 3 and suit[0] in cards:
                return suit + [suit[0]], False

    if not self_turn:  # 其他玩家回合 #明杠，吃碰
        # 计算操作前评估值
        if cards.count(op_card) == 3:
            return [op_card, op_card, op_card, op_card], False

        if len(c_cards) >= 3: # 定缺牌超过3张时，扩展无废牌报错 TODO 后续考虑向听数变化
            if op_card not in c_cards:
                return [op_card, op_card, op_card], False
            else:
                return [], False
        cards_pre = copy.copy(cards)
        # cards_pre.append(-1) #加入一张0作为下次摸到的牌，并提升一定的概率a
        score_dict_pre, min_xts_pre = get_score_dict_op(cards_pre, suits)
        if score_dict_pre != {}:
            score_pre = max(score_dict_pre.values())
        else:
            score_pre = 0
        # 计算操作后的评估值
        # 确定可选动作
        op_sets = []
        if cards.count(op_card) >= 2:
            op_sets.append([op_card, op_card])

        score_set = []
        for op_set in op_sets:
            cards_ = copy.copy(cards)
            cards_.remove(op_set[0])
            cards_.remove(op_set[1])

            suits_ = MJ.deepcopy(suits)
            suits_.append(sorted(op_set + [op_card]))

            score_dict, _ = get_score_dict_op(cards=cards_, suits=suits_, max_xts=min_xts_pre)
            # max_discard = max(score_dict, key=lambda x: score_dict[x])
            # print "score_dict",score_dict
            if score_dict != {}:
                score = max(score_dict.values())
                score_set.append(score)
        if time.time() - TIME_START > 3:
            logger.warning("op time out %s", time.time() - TIME_START)
        # print("score_set",score_set)
        if score_set == []:
            return [], False
        else:
            max_score = max(score_set)
            print("score",max_score,"score_pre", score_pre)
            if max_score > score_pre:  # TODO 测试下*1.1
                return sorted(op_sets[score_set.index(max_score)] + [op_card]), False
    return [], False



def choose_hu(handcards, suits, discards, discards_op, win_player_num,remain_num, op_card ,hu_cards = [], zi_mo = False,):
    """
    弃胡处理部分：根据手牌，定缺花色，已胡玩家（轻），番型，可胡牌张数来确定是否胡牌
    K*fan*-胡牌张数-*ln(e+人数)/8
    info = [xts,fan,hu-num,remain_num]  xts是特殊花色
    """

    if len(handcards) % 3 == 2:
        zi_mo = True
    if not zi_mo:
        handcards.append(op_card)
    handcards_1 = copy.copy(handcards)
    start = time.time()
    qys_cs = Qingyise(cards=handcards, suits=suits).qingyise_CS()
    end = time.time()
    print("qys_cs_cost",end-start)
    pph_cs = Pengpenghu(cards=handcards, suits=suits).pengpenghu_CS()
    djy_cs = Duanyaojiu(cards=handcards, suits=suits).duanyaojiu_CS()
    qd_cs = Qidui(cards=handcards,suits=suits).qidui_CS()
    xts_list = [qys_cs[0][-3], djy_cs[0][-3], qd_cs[-1]] # 取消掉pph
    min_xts = min(xts_list)  # 各特殊牌型最小xts
    #print("zimo",zi_mo)
    #print("remain_num",remain_num)
    print("qys,,djy,qd",xts_list)
    if min_xts == 0 or min_xts == 1:  # 已经是特殊番型
        return True
    handcards = handcards_1
    if not zi_mo:
        handcards.remove(op_card)  # 换成13张
        ph_cs = MJ.PingHu(cards=handcards, suits=suits, fan_type=0).pinghu_CS()
        print(ph_cs[0])
        ef_cards = []
        for cs in ph_cs:
            if len(cs[-5]) == 2:  # 两个对子
                ef_cards.append(cs[-5][0][0])
                ef_cards.append(cs[-5][1][0])
            elif len(cs[-4]) == 1:  # 搭子
                tmp = MJ.get_effective_cards(cs[-4][0])
                for card in tmp:
                    ef_cards.append(card)
            else:  #
                ef_cards.append(cs[-2][0])
        ef_cards = list(set(ef_cards))
        ef_cards_leftNum = {}

        discards_map = {0x01: 0, 0x02: 1, 0x03: 2, 0x04: 3, 0x05: 4, 0x06: 5, 0x07: 6, 0x08: 7, 0x09: 8,
                        0x11: 9, 0x12: 10, 0x13: 11, 0x14: 12, 0x15: 13, 0x16: 14, 0x17: 15, 0x18: 16, 0x19: 17,
                        0x21: 18, 0x22: 19, 0x23: 20, 0x24: 21, 0x25: 22, 0x26: 23, 0x27: 24, 0x28: 25, 0x29: 26}
        left_num, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=handcards,hu_cards=hu_cards)
        for card in ef_cards:
            ef_cards_leftNum[card] = left_num[discards_map[card]]
        win_num = sum(ef_cards_leftNum.values())  # 得到可胡牌总数
        #print("win_num",win_num)
        handcards.append(op_card)  # 换成14张，计算番

    if remain_num > 35 and (remain_num / 4) / (min_xts) > 3:  # 当剩余round/xts>3时，可以试错
        return False
    else:
        return True

    #     res = fan * math.sqrt(win_num + 1) * math.log(math.exp(1) + win_player_num) / 4
    # else:
    #     res = 1.5 * fan * math.sqrt(win_num + 1) * math.log(math.exp(1) + win_player_num) / 4
    #     print("gg")
    # if zi_mo:
    #     res = res * 1.5
    # if res>=1:
    #     print(res)
    #     print("fan:",fan,"win_num:",win_num,"zimo:",zi_mo)
    #     return True
    # else:
    #     print('0---------')
    #     print(res)
    #     print("fan:", fan, "win_num:", win_num, "zimo:", zi_mo)
    #     return False