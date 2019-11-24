import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


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


vgg = models.vgg16_bn()
pre = torch.load('../vgg16_bn-6c64b313.pth')
vgg.load_state_dict(pre)

convLayerCnt = 1
bnLayerCnt = 1
filters, channels, kernelWidth, kernelHeight = 0, 0, 0, 0
paraCnt = np.zeros(11)

for f in vgg.features:
    if isinstance(f, nn.Conv2d):
        print(convLayerCnt, '\t', f)
        filters, channels, kernelWidth, kernelHeight = \
            f.weight.size()[0], f.weight.size()[1], f.weight.size()[2], f.weight.size()[3]
        for i in range(0, filters):
            curFilter1d = f.weight.data[i].view(channels * 9).numpy()
            for j in curFilter1d:
                powerResTemp = powerRes(j)
                paraCnt[powerResTemp] += 1
        total = filters * channels * kernelWidth * kernelHeight
        paraCnt /= total
        plt.plot(np.arange(-10, 1), paraCnt[0:11])
        plt.ylabel('percentage')
        plt.xlabel('parament magnitude/lg')
        plt.title('第' + str(convLayerCnt) + '个卷积层卷积核参数分布PDF', FontProperties=font)
        plt.savefig("./fig/" + str(convLayerCnt) + "_conv_pdf.jpg")
        plt.close()
        for i in range(1, len(paraCnt)):
            paraCnt[i] += paraCnt[i - 1]
        plt.plot(np.arange(-10, 1), paraCnt[0:11])
        plt.ylabel('percentage')
        plt.xlabel('parament magnitude/lg')
        plt.title('第' + str(convLayerCnt) + '个卷积层卷积核参数分布CDF', FontProperties=font)
        plt.savefig("./fig/" + str(convLayerCnt) + "_conv_cdf.jpg")
        plt.close()
        convLayerCnt += 1
    elif isinstance(f, nn.BatchNorm2d):
        print(bnLayerCnt, '\t', f)
        filters = f.weight.size()[0]
        tmp = f.weight.data.view(filters).numpy()
        for i in tmp:
            powerResTemp = powerRes(i)
            paraCnt[powerResTemp] += 1
        paraCnt /= filters
        plt.plot(np.arange(-10, 1), paraCnt[0:11])
        plt.ylabel('percentage')
        plt.xlabel('parament magnitude/lg')
        plt.title('第' + str(convLayerCnt) + '个BN层γ参数分布PDF', FontProperties=font)
        plt.savefig("./fig/" + str(convLayerCnt) + "_γ_pdf.jpg")
        plt.close()
        for i in range(1, len(paraCnt)):
            paraCnt[i] += paraCnt[i - 1]
        plt.plot(np.arange(-10, 1), paraCnt[0:11])
        plt.ylabel('percentage')
        plt.xlabel('parament magnitude/lg')
        plt.title('第' + str(convLayerCnt) + '个BN层γ参数分布CDF', FontProperties=font)
        plt.savefig("./fig/" + str(convLayerCnt) + "_γ_cdf.jpg")
        plt.close()
        bnLayerCnt += 1
    paraCnt = np.zeros(11)

# a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [3.0, 3.0, 3.0]])
# b = torch.tensor([[2.0, 4.0, 6.0], [4.0, 5.0, 6.0], [-3.0, -3.0, -3.0]])
# print(torch.cosine_similarity(a, b, dim=1))
# a.resize_(9)
# b.resize_(9)
# print(torch.cosine_similarity(a, b, dim=0))
# print(torch.cosine_similarity(vgg.features[0].weight.data[0][0].view(9), vgg.features[0].weight.data[0][1].view(9), dim=0))
