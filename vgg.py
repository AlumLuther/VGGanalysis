import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


# 计算参数的数量级
def powerRes(myNum):
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
def plotAndSave(w, title, filename):
    plt.plot(np.arange(-10, 1), w)
    plt.ylabel('percentage')
    plt.xlabel('parament magnitude/lg')
    plt.title(title, FontProperties=font)
    plt.savefig(filename)
    plt.close()


vgg = models.vgg16_bn()
pre = torch.load('../vgg16_bn-6c64b313.pth')
vgg.load_state_dict(pre)

layerCnt = 1
paraCnt = np.zeros(11)
paraCntL1 = np.zeros(11)
sns.set()

for f in vgg.features:
    if isinstance(f, nn.Conv2d):
        print(layerCnt, '\t', f)
        filterCnt, channelCnt, kernelWidth, kernelHeight = f.weight.size()[0], f.weight.size()[1], f.weight.size()[2], f.weight.size()[3]
        filterSize = channelCnt * kernelWidth * kernelHeight
        totalSize = filterCnt * filterSize

        # 将一层中的所有卷积核展平，成为二维数组
        allFilter1d = f.weight.data.view(filterCnt, filterSize).numpy()
        # 统计同一层之间卷积核相关系数
        allFilterSimilarity = np.corrcoef(allFilter1d)
        # 统计卷积核的权重以及L1范数的分布
        for i in range(0, filterCnt):
            L1Norm = 0
            for j in range(0, filterSize):
                curWeight = allFilter1d[i][j]
                L1Norm += abs(curWeight)
                powerResTemp = powerRes(curWeight)
                paraCnt[powerResTemp] += 1
            powerResTemp = powerRes(L1Norm)
            paraCntL1[powerResTemp] += 1
        paraCnt /= totalSize
        paraCntL1 /= filterCnt
        layerCntStr = str(layerCnt)
        # 绘制相关系数热图
        ax = sns.heatmap(allFilterSimilarity)
        plt.title("第" + layerCntStr + "层（卷积层）卷积核相关系数热图", FontProperties=font)
        plt.savefig("./fig/" + layerCntStr + "_conv_similarity_heatmap.jpg")
        plt.close()
        # 绘制参数及L1范数分布图
        plotAndSave(paraCnt, "第" + layerCntStr + "层（卷积层）卷积核参数分布PDF", "./fig/" + layerCntStr + "_conv_para_pdf.jpg")
        plotAndSave(paraCntL1, "第" + layerCntStr + "层（卷积层）卷积核L1范数分布PDF", "./fig/" + layerCntStr + "_conv_L1norm_pdf.jpg")
        for i in range(1, len(paraCnt)):
            paraCnt[i] += paraCnt[i - 1]
            paraCntL1[i] += paraCntL1[i - 1]
        plotAndSave(paraCnt, "第" + layerCntStr + "层（卷积层）卷积核参数分布CDF", "./fig/" + layerCntStr + "_conv_para_cdf.jpg")
        plotAndSave(paraCntL1, "第" + layerCntStr + "层（卷积层）卷积核L1范数分布CDF", "./fig/" + layerCntStr + "_conv_L1norm_cdf.jpg")
    elif isinstance(f, nn.BatchNorm2d):
        print(layerCnt, '\t', f)
        weightCnt = f.weight.size()[0]
        tmp = f.weight.data.view(weightCnt).numpy()
        for i in tmp:
            powerResTemp = powerRes(i)
            paraCnt[powerResTemp] += 1
        paraCnt /= weightCnt
        # 绘制参数分布图
        plotAndSave(paraCnt, "第" + layerCntStr + "层（BN层）γ参数分布PDF", "./fig/" + layerCntStr + "_γ_pdf.jpg")
        for i in range(1, len(paraCnt)):
            paraCnt[i] += paraCnt[i - 1]
        plotAndSave(paraCnt, "第" + layerCntStr + "层（BN层）γ参数分布CDF", "./fig/" + layerCntStr + "_γ_cdf.jpg")
    layerCnt += 1
    paraCnt = np.zeros(11)
    paraCntL1 = np.zeros(11)
