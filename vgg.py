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

layerCnt = 1
filters, channels, kernelWidth, kernelHeight = 0, 0, 0, 0
paraCnt = np.zeros(11)

for f in vgg.features:
    if isinstance(f, nn.Conv2d):
        print(layerCnt, '\t', f)
        filters, channels, kernelWidth, kernelHeight = \
            f.weight.size()[0], f.weight.size()[1], f.weight.size()[2], f.weight.size()[3]
        print(f.weight.size())
        for i in range(0, filters):
            for j in range(0, channels):
                for k in range(0, kernelWidth):
                    for l in range(0, kernelHeight):
                        powerResTemp = powerRes(f.weight.data[i][j][k][l].item())
                        paraCnt[powerResTemp] += 1
        plt.plot(np.arange(-10, 1), paraCnt[0:11])
        plt.ylabel('计数/个', FontProperties=font)
        plt.xlabel('参数的数量级/10为底对数', FontProperties=font)
        plt.title('第' + str(layerCnt) + '层卷积核参数分布图', FontProperties=font)
        plt.savefig("./fig/"+str(layerCnt)+".jpg")
        plt.close()
        layerCnt += 1
        paraCnt = np.zeros(11)

# a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [3.0, 3.0, 3.0]])
# b = torch.tensor([[2.0, 4.0, 6.0], [4.0, 5.0, 6.0], [-3.0, -3.0, -3.0]])
# print(torch.cosine_similarity(a, b, dim=1))
# a.resize_(9)
# b.resize_(9)
# print(torch.cosine_similarity(a, b, dim=0))

# print(vgg.features[0].weight.data)
print(torch.cosine_similarity(vgg.features[0].weight.data[0][0].view(9), vgg.features[0].weight.data[0][1].view(9), 0))
