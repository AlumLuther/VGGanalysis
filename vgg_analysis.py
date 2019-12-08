import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


# 计算参数的数量级
def power_res(myNum):
    if myNum < 0:
        myNum = -myNum
    res = 10
    if myNum < 1:
        if myNum < 1e-10:
            res = 0
            return res
        while myNum < 1:
            myNum *= 10
            res -= 1
    return res


# 绘制并保存折线图
def plot_and_save(w, title, filename):
    plt.plot(np.arange(-10, 1), w)
    plt.ylabel('percentage')
    plt.xlabel('parament magnitude/lg')
    plt.title(title, FontProperties=font)
    plt.savefig(filename)
    plt.close()


vgg = models.vgg16_bn()
pre = torch.load('../vgg16_bn-6c64b313.pth')
vgg.load_state_dict(pre)

layer_cnt = 1
conv_layer_cnt = 1
bn_layer_cnt = 1
para_cnt = np.zeros(11)
para_cnt_l1 = np.zeros(11)
sns.set()
os.makedirs("./fig/conv/corrcoef/")
os.makedirs("./fig/conv/parameter/pdf/")
os.makedirs("./fig/conv/parameter/cdf/")
os.makedirs("./fig/conv/l1norm/pdf/")
os.makedirs("./fig/conv/l1norm/cdf/")
os.makedirs("./fig/bn/gamma/pdf/")
os.makedirs("./fig/bn/gamma/cdf/")

for f in vgg.features:
    if isinstance(f, nn.Conv2d):
        print(layer_cnt, '\t', f)
        filter_cnt, channel_cnt, kernel_width, kernel_height = f.weight.size()[0], f.weight.size()[1], f.weight.size()[2], f.weight.size()[3]
        filter_size = channel_cnt * kernel_width * kernel_height
        total_cnt = filter_cnt * filter_size
        kernelCnt = filter_cnt * channel_cnt
        kernel_size = kernel_width * kernel_height
        # 将一层中的所有卷积核展平，成为二维数组
        filters_space = f.weight.data.view(filter_cnt, filter_size).numpy()
        # 统计同一层之间卷积核相关系数
        all_filter_similarity = list(map(abs, np.corrcoef(filters_space)))
        # 统计卷积核的权重以及L1范数的数量级分布
        for i in range(0, filter_cnt):
            L1Norm = 0
            for j in range(0, filter_size):
                curWeight = filters_space[i][j]
                L1Norm += abs(curWeight)
                power_resTemp = power_res(curWeight)
                para_cnt[power_resTemp] += 1
                if (j + 1) % kernel_size == 0:
                    power_resTemp = power_res(L1Norm)
                    para_cnt_l1[power_resTemp] += 1
                    L1Norm = 0
        para_cnt /= total_cnt
        para_cnt_l1 /= kernelCnt
        # 绘制相关系数热图
        ax = sns.heatmap(all_filter_similarity)
        plt.title("第" + str(conv_layer_cnt) + "卷积层卷积核相关系数热图", FontProperties=font)
        plt.savefig("./fig/conv/corrcoef/" + str(conv_layer_cnt) + ".jpg")
        plt.close()
        # 绘制参数及L1范数分布图
        plot_and_save(para_cnt, "第" + str(conv_layer_cnt) + "卷积层卷积核参数分布PDF", "./fig/conv/parameter/pdf/" + str(conv_layer_cnt) + ".jpg")
        plot_and_save(para_cnt_l1, "第" + str(conv_layer_cnt) + "卷积层卷积核L1范数分布PDF", "./fig/conv/l1norm/pdf/" + str(conv_layer_cnt) + ".jpg")
        for i in range(1, len(para_cnt)):
            para_cnt[i] += para_cnt[i - 1]
            para_cnt_l1[i] += para_cnt_l1[i - 1]
        plot_and_save(para_cnt, "第" + str(conv_layer_cnt) + "卷积层卷积核参数分布CDF", "./fig/conv/parameter/cdf/" + str(conv_layer_cnt) + ".jpg")
        plot_and_save(para_cnt_l1, "第" + str(conv_layer_cnt) + "卷积层卷积核L1范数分布CDF", "./fig/conv/l1norm/cdf/" + str(conv_layer_cnt) + ".jpg")
        conv_layer_cnt += 1
    elif isinstance(f, nn.BatchNorm2d):
        print(layer_cnt, '\t', f)
        weight_cnt = f.weight.size()[0]
        tmp = f.weight.data.view(weight_cnt).numpy()
        for i in tmp:
            power_resTemp = power_res(i)
            para_cnt[power_resTemp] += 1
        para_cnt /= weight_cnt
        # 绘制参数分布图
        plot_and_save(para_cnt, "第" + str(bn_layer_cnt) + "BN层γ参数分布PDF", "./fig/bn/gamma/pdf/" + str(bnlayer_cnt) + "jpg")
        for i in range(1, len(para_cnt)):
            para_cnt[i] += para_cnt[i - 1]
        plot_and_save(para_cnt, "第" + str(bn_layer_cnt) + "BN层γ参数分布CDF", "./fig/bn/gamma/cdf/" + str(bnlayer_cnt) + ".jpg")
        bn_layer_cnt += 1
    layer_cnt += 1
    para_cnt = np.zeros(11)
    para_cnt_l1 = np.zeros(11)
