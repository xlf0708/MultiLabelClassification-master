import copy
import time

import numpy
import numpy as np

from ..SCMJ import sichuanMJ_v2 as SCMJ


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import json
import random
import sys

# path = "../data/recommond_2022425.txt"
# def data_handle():
#     f = open(path, 'r', encoding='utf-8')
#     data = f.readlines()  #
#     for i in range(len(data)):  # 113225
#         json_data = eval(data[i])
#
#         features, label_numpy=calculate_king_sys_suphx(json_data)



def cal_xts(hand_cards,operate_cards):
    """
    计算xts及相对应权重


    :return: list——xts
    """


    PH = SCMJ.SearchTree_PH(cards=copy.copy(hand_cards), suits=copy.copy(operate_cards), padding=[])
    QYS = SCMJ.Qingyise(cards=copy.copy(hand_cards), suits=copy.copy(operate_cards), padding=[])
    PPH = SCMJ.Pengpenghu(cards=copy.copy(hand_cards), suits=copy.copy(operate_cards), padding=[])
    DYJ = SCMJ.Duanyaojiu(cards=copy.copy(hand_cards), suits=copy.copy(operate_cards), padding=[])
    CS_PH = PH.pinghu_CS()
    CS_QYS = QYS.qingyise_CS()
    CS_PPH = PPH.pengpenghu_CS()
    CS_DYJ = DYJ.duanyaojiu_CS()
    xts_list = [CS_PH[0][-3], CS_QYS[0][-3], CS_PPH[-1], CS_DYJ[0][-3]]
    val = 0
    # print("xts",xts_list)
    # 取3个特殊牌型中较小2个
    max_index= xts_list.index(max(xts_list[1:]))
    # for i in range(1,4):
    #     if(i==1):
    #         val+=i//4
    #     else:
    #         val+=i//2
    #
    for i in range(1,4):
        if( i== max_index):
            continue
        elif(xts_list[i] == 14):
            val += 7
        else:
            val+=xts_list[i]
    return xts_list[0], val

def transToNum(op_card):
    """
    四川麻将0-26
    :param op_card:
    :return:
    """

    if op_card >= 1 and op_card <= 9:
        op_card = op_card - 1
    elif op_card >= 17 and op_card <= 25:
        op_card = op_card - 8
    elif op_card >= 33 and op_card <= 41:
        op_card = op_card - 15
    return op_card
def suphx_cards_feature_code(cards_, channels):
    """

    :param cards_: 一维数组手牌
    :param channels: 默认4
    :return:  返回27*4矩阵（四川）
    """
    cards = copy.deepcopy(cards_)
    if not isinstance(cards, list):  # 如果是一张牌
        cards = [cards]
    features = []
    for channel in range(channels):
        S = set(cards)
        feature = [0] * 27
        for card in S:
            card_index = transToNum(card)# 16进制转化成0-26的形式
            cards.remove(card)
            feature[card_index] = 1
        features.append(feature)
    return features

'''
@msg: 返回对数据按数据类型编码的特征
@param {*} data
@param {*} channels
@param {*} data_type  数据类型  optional ["cards_set", "seq_discards", "dummy"]
@return {*}
'''
def suphx_data_feature_code(data, channels=4, data_type="cards_set"):
    # cards 为16进制
    data_copy = copy.deepcopy(data)
    features = []
    if data_type == "cards_set": # 手牌
        features.extend(suphx_cards_feature_code(data_copy, channels))
    elif data_type == "seq_discards": # 弃牌保证顺序，一个通道
        seq_discards_features = []  # 弃牌的features,四个玩家的弃牌顺序，
        seq_len = 25  # 每个玩家弃牌的最大手数为25手  # ! resnet50x_MT_jsondataset版本 25
        for player_discard_seq in data_copy:
            cur_seq_discards_features = []  # 当前玩家的弃牌序列
            for i in range(len(player_discard_seq)):
                if i >= 25:
                    break
                cur_seq_discards_features.extend(suphx_cards_feature_code(player_discard_seq[i], channels))

            seq_discards_features.extend(cur_seq_discards_features)  # 把当前已有的序列添加到features中
            need_pad_len = seq_len - len(cur_seq_discards_features)  #需要填充的长度

            pad_features = [[0]*27 for _ in range(need_pad_len)]
            seq_discards_features.extend(pad_features)
        features.extend(seq_discards_features)
    elif data_type == "seq_discards_one_people":
        seq_len = 25  # !每个玩家弃牌的最大手数为30手
        cur_seq_discards_features = []  # 当前玩家的弃牌序列
        for i in range(len(data_copy)):
            if i >= 25:
                break
            cur_seq_discards_features.extend(suphx_cards_feature_code(data_copy[i], channels))

        features.extend(cur_seq_discards_features)  # 把当前已有的序列添加到features中
        need_pad_len = seq_len - len(cur_seq_discards_features)  #需要填充的长度

        pad_features = [[0]*27 for _ in range(need_pad_len)]
        features.extend(pad_features)
    elif data_type == "dummy":  # 哑变量编码  此时的data为整数
        assert isinstance(data_copy, int)
        dummy_features = [[0]*27 for _ in range(channels)]
        if 0 < data_copy <= channels:
            dummy_features[data_copy - 1] = [1] * 27
        elif data_copy == 0:
            # pass  当为0时，哑变量全为零
            pass
        else:
            print(data_copy, channels)
            print(data)
            print("INFO[ERROR]")
            pass
        features.extend(dummy_features)
    elif data_type == "dummy_v2":  # 哑变量编码  “0”也进行编码
        assert isinstance(data_copy, int)
        dummy_features = [[0]*34 for _ in range(channels)]
        if 0 <= data_copy <= channels:
            dummy_features[data_copy] = [1] * 34
        else:
            print(data_copy, channels)
            print("dummy_v2 INFO[ERROR]")
            pass
        features.extend(dummy_features)
    elif data_type == "look_ahead":  # 暂时空着
        pass

    return features

# 模仿suphx对牌进行编码
# sc
def calculate_king_sys_suphx(state_info_json, label_type='wt'):
    """
    牌都是用16进制进行表示，参数需要预先处理好
    输入数据类型
    {'seat_id': 1,
    'dealer_id': 1,
    'catch_card': 40,
    'user_cards': {'hand_cards': [7, 8, 9, 17, 19, 21, 23, 24, 24, 25, 35, 37, 38, 40], 'operate_cards': []},
    'discards': [[], [], [], []],
    'discards_real': [[], [], [], []],
    'discards_op': [[], [], [], []],
    'hu_cards': [[], [], [], []],
    'colors': [0, 2, 2, 0],
    'round': 1,
    'remain_num': 55,
    'wall': [3, 41, 5, 8, 17, 39, 7, 40, 22, 22, 18, 40, 18, 33, 1, 40, 8, 4, 2, 1, 39, 33, 22, 34, 5, 38, 35, 2, 4, 21, 6,
          17, 41, 34, 38, 21, 7, 23, 24, 5, 36, 6, 37, 24, 33, 22, 37, 6, 25, 38, 7, 35, 9, 9, 37],
    'hands': [[18, 19, 19, 25, 25, 34, 34, 35, 36, 41, 33, 41, 36], [17, 19, 21, 23, 24, 24, 25, 35, 37, 38, 40, 9, 8, 7],
           [2, 2, 3, 4, 5, 6, 17, 18, 20, 20, 1, 9, 3], [1, 3, 4, 8, 19, 20, 20, 21, 23, 23, 36, 39, 39]]}
    """

    """ 
    基本数据结构(vector)： 34 维
    我的手牌：  4（vector）
    4家出牌：   25*4      最多25个轮次
    4家副露：   4*4*4       最多4个副露，每个副露最多四张牌
    宝牌：     1
    牌墙数：     84         最多84张牌
    当前轮数：   25         最多25个轮次
    4家飞宝数：   4*4       4家最多飞4宝
    庄家id        4         最多4个玩家 
    """
    # print(state_info_json)
    seat_id = state_info_json["seat_id"]
    handCards4= state_info_json["hands"] # 四人手牌
    handCards0 = state_info_json["user_cards"]["hand_cards"] # 自己手牌
    # print("sjoupai",handCards4)
    fulu_ = state_info_json["discards_op"]
    # print("fulu",fulu_)
    colors =  state_info_json["colors"]
    discards = state_info_json["discards"]
    discards_real = state_info_json["discards_real"]
    remain_num =  state_info_json["remain_num"]
    # wall = state_info_json["wall"]
    round_ =  state_info_json["round"]
    label = state_info_json["lable"]
    # "lable": [39, 39, 30, 39]
    # print(label)
    # 所有特征
    features = []

    # 自己手牌特征
    handcards_features = suphx_data_feature_code(handCards0, 4)
    # print("handfeatures",handcards_features)
    features.extend(handcards_features)
    # 四人手牌特征
    handcards4_features = []
    for handcards in handCards4:
        handcards_features = suphx_data_feature_code(handcards,4)
        handcards4_features.extend(handcards_features)
    features.extend(handcards4_features)
    # 副露特征
    fulu_features = []
    for fulu in fulu_:
        action_features = []
        fulu_len = len(fulu)  # 当前玩家副露的长度
        for action in fulu:
            action_features.extend(suphx_data_feature_code(action, 4))
        # 需要padding
        action_padding_features = [[0] * 27 for _ in range(4) for _ in range(4 - fulu_len)]
        action_features.extend(action_padding_features)

        fulu_features.extend(action_features)
    # print("fulu_f", fulu_features)
    features.extend(fulu_features)

    #所有弃牌的    顺序信息
    seq_discards_features = suphx_data_feature_code(discards, 1, data_type="seq_discards")
    features.extend(seq_discards_features)

    #所有弃牌的    顺序信息
    seq_discards_real_features = suphx_data_feature_code(discards_real, 1, data_type="seq_discards")
    features.extend(seq_discards_real_features)
    # 剩余牌数特征
    remian_cardsnums_features = suphx_data_feature_code(remain_num, 55, data_type="dummy")
    features.extend(remian_cardsnums_features)
    # seatID
    seat_id_feature = suphx_data_feature_code(seat_id,4,data_type="dummy")
    features.extend(seat_id_feature)
    # 当前手数
    cur_round_features = suphx_data_feature_code(round_, 25, data_type="dummy")
    features.extend(cur_round_features)

    # 花色
    colors_features = []
    for color in colors:
        color_features = suphx_data_feature_code(color,3,data_type="dummy")
        colors_features.extend(color_features)
    features.extend(colors_features)
    # !label
    #3 15
    # label = []
    # # for i in range(4):
    # #     min_xts, sum_xts = cal_xts(state_info_json["hands"][i],
    # #                                state_info_json["discards_op"][i])
    # #     label.append(28-min_xts-sum_xts)
    # # print(label)  #四人牌力评估# 28-(min+plus)
    # label_ = []
    num = 0
    for i in range(4):
        if i==seat_id:
            continue
        else:
            if(label[i] > label[seat_id]):
                num+=1
    if label_type == 'wt':  # 听牌标签
        label_numpy = np.zeros(4)
        label_numpy[num] = 1  # 1 *4[0.0.0]

    # feature 维度转换
    features = np.array(features).astype(np.float32)
    features = features.T
    features = np.expand_dims(features, 0)
    features = features.transpose([2, 1, 0]) # 更换位置  转换成c × 27 × 1的格式
    # print(type(features))
    # features = np.expand_dims(features, 0) # 1 x c x 34 x 1
    # print(type(features))
    # print(features.shape)
    # print(label_numpy.shape)
    return features, label_numpy

if __name__ == '__main__':
   pass

