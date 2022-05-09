# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
# from torchsummary import summary

# from model.conv_lstm import ConvLSTM

resnet_dic = {"resnet18": models.resnet18,
              "resnet34": models.resnet34,
              "resnet50": models.resnet50,
              "resnet101": models.resnet101,
              "resnet152": models.resnet152,
              }


def oppHead(in_planes, places):
    return nn.Sequential(
        nn.Linear(in_planes, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512, eps=1e-2),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256, eps=1e-2),
        nn.Linear(256, places)
        # return nn.Sequential(
        #     nn.Linear(in_planes, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(2048, eps=1e-2),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(1024, eps=1e-2),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(512, eps=1e-2),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(256, eps=1e-2),
        #     nn.Linear(256, places)
    )


# --------------------------------------------------------------------------
# 1. LSTM model
# 数据结构：4*6*34*34     前几手 channel height weight 
# class LSTM_multitask_model(torch.nn.Module):
#     def __init__(self, args):
#         super(LSTM_multitask_model, self).__init__()
#         self.conv_lstm1 = ConvLSTM(input_dim=6, hidden_dim=[10], kernel_size=(3, 3),
#                                    num_layers=1, batch_first=True, bias=True, return_all_layers=False)
#         # self.bn1 = nn.BatchNorm2d(10, eps=2e-1)
#         self.conv_lstm2 = ConvLSTM(input_dim=10, hidden_dim=[5], kernel_size=(3, 3),
#                                    num_layers=1, batch_first=True, bias=True, return_all_layers=False)
#         # self.bn2 = nn.BatchNorm2d(5, eps=2e-1)
#         self.conv_lstm3 = ConvLSTM(input_dim=5, hidden_dim=[3], kernel_size=(3, 3),
#                                    num_layers=1, batch_first=True, bias=True, return_all_layers=False)
#         # self.bn3 = nn.BatchNorm2d(3, eps=2e-1)
#         self.conv_lstm4 = ConvLSTM(input_dim=3, hidden_dim=[1], kernel_size=(3, 3),
#                                    num_layers=1, batch_first=True, bias=True, return_all_layers=False)
#         # self.bn4 = nn.BatchNorm2d(1, eps=2e-1)
#
#         # self.d_out = nn.Dropout(0.3)
#         # # opp1
#         # self.y1_fc1 = nn.Linear(34*34*4, 1024)
#         # self.y1_bn1 = nn.BatchNorm1d(1024, eps=1e-2)
#
#         # self.y1_fc2 = nn.Linear(1024, 512)
#         # self.y1_bn2 = nn.BatchNorm1d(512, eps=1e-2)
#
#         # self.y1_fc3 = nn.Linear(512, 256)
#         # self.y1_bn3 = nn.BatchNorm1d(256, eps=1e-2)
#
#         # self.y1_head = nn.Linear(256, 2)
#
#         # # opp2
#         # self.y2_fc1 = nn.Linear(34*34*4, 1024)
#         # self.y2_bn1 = nn.BatchNorm1d(1024, eps=1e-2)
#
#         # self.y2_fc2 = nn.Linear(1024, 512)
#         # self.y2_bn2 = nn.BatchNorm1d(512, eps=1e-2)
#
#         # self.y2_fc3 = nn.Linear(512, 256)
#         # self.y2_bn3 = nn.BatchNorm1d(256, eps=1e-2)
#
#         # self.y2_head = nn.Linear(256, 2)
#
#         # # opp3
#         # self.y3_fc1 = nn.Linear(34*34*4, 1024)
#         # self.y3_bn1 = nn.BatchNorm1d(1024, eps=1e-2)
#
#         # self.y3_fc2 = nn.Linear(1024, 512)
#         # self.y3_bn2 = nn.BatchNorm1d(512, eps=1e-2)
#
#         # self.y3_fc3 = nn.Linear(512, 256)
#         # self.y3_bn3 = nn.BatchNorm1d(256, eps=1e-2)
#
#         # self.y3_head = nn.Linear(256, 2)
#         self.y1o = oppHead(34*34*4, 2)
#         self.y2o = oppHead(34*34*4, 2)
#         self.y3o = oppHead(34*34*4, 2)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#
#     def forward(self, input):
#         # 4个lstm_conv层
#         x = self.conv_lstm1(input)[0][0]
#         x = self.conv_lstm2(x)[0][0]
#         x = self.conv_lstm3(x)[0][0]
#         x = self.conv_lstm4(x)[0][0]
#
#         # flatten
#         x = x.view(x.size(0), -1)
#
#         # # #opp1后面接若干全连接层
#         # y1o = self.y1_bn1(F.relu(self.y1_fc1(x)))
#         # y1o = self.y1_bn2(F.relu(self.y1_fc2(y1o)))
#         # y1o = self.y1_bn3(F.relu(self.y1_fc3(y1o)))
#         # y1o = F.softmax(self.y1_head(y1o),dim=1)
#
#         # # #opp2后面接若干全连接层
#         # y2o = self.y2_bn1(F.relu(self.y2_fc1(x)))
#         # y2o = self.y2_bn2(F.relu(self.y2_fc2(y2o)))
#         # y2o = self.y2_bn3(F.relu(self.y2_fc3(y2o)))
#         # y2o = F.softmax(self.y2_head(y2o),dim=1)
#         # # #opp3后面接若干全连接层
#         # y3o = self.y3_bn1(F.relu(self.y3_fc1(x)))
#         # y3o = self.y3_bn2(F.relu(self.y3_fc2(y3o)))
#         # y3o = self.y3_bn3(F.relu(self.y3_fc3(y3o)))
#         # y3o = F.softmax(self.y3_head(y3o),dim=1)
#         y1o = F.softmax(self.y1o(x), dim=1)
#         y2o = F.softmax(self.y2o(x), dim=1)
#         y3o = F.softmax(self.y3o(x), dim=1)
#
#         # print(y1o.shape)
#         return y1o, y2o, y3o


# -------------------------------------------------------------------------
# 2. Simple_multitask_model
# 数据结构 56*34*4
# simple model
def Conv(in_planes, places, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places,
                  kernel_size=(5, 2), stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True)
    )


class Simple_multitask_model(torch.nn.Module):
    def __init__(self, args):
        super(Simple_multitask_model, self).__init__()
        self.conv1 = Conv(in_planes=56, places=100, stride=1)
        self.d_out1 = nn.Dropout(0.5)
        self.conv2 = Conv(in_planes=100, places=100, stride=1)
        self.d_out2 = nn.Dropout(0.5)
        self.conv3 = Conv(in_planes=100, places=100, stride=1)
        self.d_out3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2200, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.d_out4 = nn.Dropout(0.5)

        self.y1o = nn.Linear(300, 2)
        self.y2o = nn.Linear(300, 2)
        self.y3o = nn.Linear(300, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        
        x = self.conv1(inputs)
        x = self.d_out1(x)
        x = self.conv2(x)
        x = self.d_out2(x)
        x = self.conv3(x)
        x = self.d_out3(x)

        # flatten
        x = x.view(x.size(0), -1)

        x = self.bn1(F.relu(self.fc1(x)))
        x = self.d_out4(x)

        y1o = F.softmax(self.y1o(x), dim=1)
        y2o = F.softmax(self.y2o(x), dim=1)
        y3o = F.softmax(self.y3o(x), dim=1)
        return y1o, y2o, y3o


# ----------------------------------------------------------
# 3. resnet model
class Resnet_multitask_model(torch.nn.Module):
    def __init__(self, args, phase='resnet50'):
        super(Resnet_multitask_model, self).__init__()
        # 使用resnet预训练模型
        model_ft = resnet_dic[phase](pretrained=True)
        # 修改输入层的通道数
        w = model_ft.conv1.weight.clone()
        model_ft.conv1 = nn.Conv2d(6, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model_ft.conv1.weight = torch.nn.Parameter(
            torch.cat((w, torch.zeros(64, 6-3, 7, 7)), dim=1))
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        if not args.requires_grad:
            for param in model_ft.parameters():
                param.requires_grad = False
            print(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 1024)

        self.resnet_feature = model_ft
        self.y1o = oppHead(1024, 2)
        self.y2o = oppHead(1024, 2)
        self.y3o = oppHead(1024, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        x = self.resnet_feature(inputs)
        y1o = F.softmax(self.y1o(x), dim=1)
        y2o = F.softmax(self.y2o(x), dim=1)
        y3o = F.softmax(self.y3o(x), dim=1)
        return y1o, y2o, y3o


def Conv1(in_planes, places, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=(
            3, 1), stride=stride, padding=(1, 0), bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True)
    )


def Conv2(in_planes, places, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=(
            3, 1), stride=stride, padding=(0, 0), bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True)
    )


def Conv_last(in_planes, places, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True)
    )


# Resnet18 34 50 101 152 
class Resnet18(Resnet_multitask_model):
    def __init__(self, args):
        super().__init__(args, phase="resnet18")


class Resnet34(Resnet_multitask_model):
    def __init__(self, args):
        super().__init__(args, phase="resnet34")


class Resnet50(Resnet_multitask_model):
    def __init__(self, args):
        super().__init__(args, phase="resnet50")


class Resnet101(Resnet_multitask_model):
    def __init__(self, args):
        super().__init__(args, phase="resnet101")


class Resnet152(Resnet_multitask_model):
    def __init__(self, args):
        super().__init__(args, phase="resnet152")

# ---------------------------------------------------------------------------
# 4. resnet_50X_MT 参考suphx 多任务任务模型
# 数据结构：298*34*1



class ResNet50X_multitask_model_base(torch.nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet50X_multitask_model_base, self).__init__()
        self.expansion = expansion
        self.num_classes = num_classes

        self.conv1 = Conv1(in_planes=303, places=256)

        self.layer = self.make_layer(
            in_places=256, places=256, blocks=blocks, stride=1)
        self.bn_2d = nn.BatchNorm2d(256, eps=2e-1)

        # 第一版的resnet50x结构
        # self.conv_last_opp1 = Conv_last(in_planes=256,places=1,stride=1)
        # self.conv_last_opp2 = Conv_last(in_planes=256,places=1,stride=1)
        # self.conv_last_opp3 = Conv_last(in_planes=256,places=1,stride=1)

        # 第二版 参考suphx的立直 吃碰杠模型结构
        self.conv2 = Conv2(in_planes=256, places=32)
        self.fc1 = nn.Linear(1024, 256)
        # nn.init.xavier_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(256)
        # heads
        self.y1o = nn.Linear(256, self.num_classes)
        self.y2o = nn.Linear(256, self.num_classes)
        self.y3o = nn.Linear(256, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def make_layer(self, in_places, places, blocks, stride):
        layers = []
        for i in range(blocks):
            layers.append(Bottleneck(in_places, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer(x)
        x = self.bn_2d(F.relu(x))

        # heads
        # 第一版结构
        # x1 = self.conv_last_opp1(x)
        # x2 = self.conv_last_opp2(x)
        # x3 = self.conv_last_opp3(x)
        # 第二版结构
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.bn1(F.relu(self.fc1(x)))

        # heads
        # 第一版
        # y1o = torch.sigmoid(x1.view(x1.size(0),-1))  # should be sigmoid
        # y2o = torch.sigmoid(x1.view(x2.size(0),-1))  # should be sigmoid
        # y3o = torch.sigmoid(x1.view(x3.size(0),-1))  # should be sigmoid
        # y1o = torch.sigmoid(self.y1o(x))  # should be sigmoid
        # y2o = torch.sigmoid(self.y2o(x))  # should be sigmoid
        # y3o = torch.sigmoid(self.y3o(x))  # should be sigmoid
        y1o = F.softmax(self.y1o(x), dim=1)
        y2o = F.softmax(self.y2o(x), dim=1)
        y3o = F.softmax(self.y3o(x), dim=1)

        return y1o, y2o, y3o


class ResNet50X_multitask_model(ResNet50X_multitask_model_base):
    def __init__(self, args):
        super(ResNet50X_multitask_model, self).__init__(50, num_classes=2)

class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=(
                3, 1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=(
                3, 1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(places)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out += residual
        out = self.relu(out)
        return out
# ---------------------------------------------------------------------------
# 5. resnet_50X 参考suphx 单任务任务模型
# 数据结构：298*34*1
class ResNet(torch.nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.num_classes = num_classes

        # self.conv1 = Conv1(in_planes=298, places=256)  # suphx
        # self.conv1 = Conv1(in_planes=116, places=256)  # 爆打
        # self.conv1 = Conv1(in_planes=146, places=256)  # 爆打 + 手切数
        # self.conv1 = Conv1(in_planes=176, places=256)  # 爆打 + 手切数 + 每手是否手切
        self.conv1 = Conv1(in_planes=380, places=256)  # 四川
        self.layer = self.make_layer(
            in_places=256, places=256, blocks=blocks, stride=1)
        self.bn_2d = nn.BatchNorm2d(256)

        # 第一版的resnet50x结构
        # self.conv_last_opp1 = Conv_last(in_planes=256,places=1,stride=1)
        # self.conv_last_opp2 = Conv_last(in_planes=256,places=1,stride=1)
        # self.conv_last_opp3 = Conv_last(in_planes=256,places=1,stride=1)

        # 第二版 参考suphx的立直 吃碰杠模型结构
        self.conv2 = Conv2(in_planes=256, places=32)
        # self.fc1 = nn.Linear(1024, 256)
        self.fc1 = nn.Linear(800, 256)  # 四川修改
        # nn.init.xavier_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(256)
        # heads 不是多任务, 所以只有一个输出头
        self.out = nn.Linear(256, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm1d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def make_layer(self, in_places, places, blocks, stride):
        layers = []
        for i in range(blocks):
            layers.append(Bottleneck(in_places, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer(x)
        x = self.bn_2d(F.relu(x))

        # heads
        # 第一版结构
        # x1 = self.conv_last_opp1(x)
        # x2 = self.conv_last_opp2(x)
        # x3 = self.conv_last_opp3(x)
        # 第二版结构
        x = self.conv2(x)
        # print("--",x.shape)
        x = x.view(x.size(0), -1)
        x = self.bn1(F.relu(self.fc1(x)))

        # heads
        # 第一版
        # y1o = torch.sigmoid(x1.view(x1.size(0),-1))  # should be sigmoid
        # y2o = torch.sigmoid(x1.view(x2.size(0),-1))  # should be sigmoid
        # y3o = torch.sigmoid(x1.view(x3.size(0),-1))  # should be sigmoid
        # out = torch.sigmoid(self.out(x))
        out = F.softmax(self.out(x), dim=1)

        return out


class ResNet50X(ResNet):
    def __init__(self, args):
        super(ResNet50X, self).__init__(50, num_classes=2)


class ResNet5X(ResNet):
    def __init__(self, args):
        super(ResNet5X, self).__init__(5, num_classes=2)


# ----------------------------------------------------------------------------
# 6. srwt_model_v5
# 数据结构 116*34*1
def Conv_v5_sample(in_planes, places, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places,
                  kernel_size=(3, 1), stride=stride, padding=(1,0), bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True)
    )


def Conv_v5_valid(in_planes, places, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places,
                  kernel_size=(3, 1), stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True)
    )


class Srwt_model_v5(torch.nn.Module):
    def __init__(self, args):
        super(Srwt_model_v5, self).__init__()
        self.conv1 = Conv_v5_sample(in_planes=116, places=256, stride=1)
        # self.conv1 = Conv_v5_sample(in_planes=146, places=256, stride=1) # 爆打 + 手切数 
        # self.conv1 = Conv_v5_sample(in_planes=176, places=256, stride=1) # 爆打 + 手切数 + 每手是否手切
        self.d_out1 = nn.Dropout(0.5)
        self.conv2 = Conv_v5_sample(in_planes=256, places=512, stride=1)
        self.d_out2 = nn.Dropout(0.5)
        self.conv3 = Conv_v5_valid(in_planes=512, places=32, stride=1)
        self.d_out3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 256)
        self.bn1 = nn.BatchNorm1d(256)#正则化。加快训练速度
        self.d_out4 = nn.Dropout(0.5)
        self.out = nn.Linear(256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        
        x = self.conv1(inputs)
        x = self.d_out1(x)
        x = self.conv2(x)
        x = self.d_out2(x)
        x = self.conv3(x)
        x = self.d_out3(x)
        # flatten
        print("--",x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.d_out4(x)
        out = F.softmax(self.out(x), dim=1)
        return out



if __name__ == "__main__":
    channels = 6
    # model = ConvLSTM(input_dim=channels,
    #              hidden_dim=[64, 64, 128],
    #              kernel_size=(3, 3),
    #              num_layers=3,
    #              batch_first=True,
    #              bias=True,
    #              return_all_layers=False)
    # input = Variable(torch.randn(10, 5, 6, 34, 34))
    # model = multi_output_model()
    # optimizer_1 = torch.optim.
    # device = torch.device("cpu")
    # model.to(device)
    # out = model(input.to(device))
    # print(out)
    # print(y2)
    # print(y3)

    # ----------------------------------------------------------------------
    # 测试Simple_multitask_model 
    # 数据结构 56*34*4
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Simple_multitask_model([])
    # print(device)
    # model = model.to(device)
    # x = Variable(torch.randn(20, 56, 34, 4))
    # y = model(x.to(device))
    # summary(model, (56, 34, 4))
    # print(list(model.named_parameters()))

    # ---------------------------------------------------------------------
    # 测试resnet 50X 参考suphx
    # 数据结构：226*34*1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Srwt_model_v5([])
    print(device)
    model = model.to(device)
    # print(model)
    x = Variable(torch.randn(256, 116, 34, 1))
    y = model(x.to(device))
    print(y.shape)
    # summary(model, (298, 34, 1))
    # print(list(model.named_parameters()))

    

