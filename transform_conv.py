import torch
import cv2
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from matplotlib.font_manager import FontProperties


def conv_to_matrix(conv):
    """
    transform  4-d filters in conv to a matrix
    :param conv: conv module
    :return: 2-d numpy array. each row is one filter.
    """
    weight = conv.weight.data
    matrix = weight.view(weight.size(0), -1).cpu().numpy()
    return matrix


def conv_dct(net):
    """
    transform all conv into frequency matrix
    :param net:
    :return:a list containing frequency matrix for each conv layer
    """
    frequency_matrix = []
    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            weight_matrix = conv_to_matrix(mod)
            frequency_matrix += [cv2.dct(weight_matrix.T)]
    return frequency_matrix


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using: " + torch.cuda.get_device_name(torch.cuda.current_device()) + " of capacity: " + str(
        torch.cuda.get_device_capability(torch.cuda.current_device())))
    '''选择网络模型'''
    net = models.vgg16_bn()
    net.load_state_dict(torch.load('../vgg16_bn-6c64b313.pth'))
    net.to(device)
    freq_matrix = conv_dct(net)


# num_tmp = 0
# for f in net.features:
#     if isinstance(f, torch.nn.Conv2d):
#         print(f.weight.size(), freq_matrix[num_tmp].shape)
#         num_tmp += 1;


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
def plotAndSave(w_space, w_freq, title, filename):
    plt.plot(np.arange(-10, 1), w_space, color="red", label='Parameters in spatial domain')
    plt.plot(np.arange(-10, 1), w_freq, color="skyblue", label='Parameters in frequency domain')
    plt.legend()
    plt.ylabel('percentage')
    plt.xlabel('parament magnitude/lg')
    plt.title(title, FontProperties=font)
    plt.savefig(filename)
    plt.close()


convLayerCnt = 1
para_cnt_space = np.zeros(11)
para_cnt_freq = np.zeros(11)
sns.set()
pdf_dir = "./fig_comp/pdf/"
cdf_dir = "./fig_comp/cdf/"
if not os.path.isdir(pdf_dir):
    os.makedirs(pdf_dir)
if not os.path.isdir(cdf_dir):
    os.makedirs(cdf_dir)
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

for f in net.features:
    if isinstance(f, torch.nn.Conv2d):
        print(convLayerCnt, '\t', f)
        filterCnt, channelCnt, kernelWidth, kernelHeight = f.weight.size()[0], f.weight.size()[1], f.weight.size()[2], f.weight.size()[3]
        filterSize = channelCnt * kernelWidth * kernelHeight
        totalCnt = filterCnt * filterSize
        allFilter1d = f.weight.data.view(filterCnt, filterSize).cpu().numpy()
        curFreqMatrix = freq_matrix[convLayerCnt - 1]
        for i in range(0, filterCnt):
            for j in range(0, filterSize):
                curWeight = allFilter1d[i][j]
                powerResTemp = powerRes(curWeight)
                para_cnt_space[powerResTemp] += 1
                curWeight = curFreqMatrix[j][i]
                powerResTemp = powerRes(curWeight)
                para_cnt_freq[powerResTemp] += 1
        para_cnt_space /= totalCnt
        para_cnt_freq /= totalCnt
        plotAndSave(para_cnt_space, para_cnt_freq, "第" + str(convLayerCnt) + "卷积层卷积核参数分布PDF", "./fig_comp/pdf/" + str(convLayerCnt) + ".jpg")
        for i in range(1, len(para_cnt_freq)):
            para_cnt_space[i] += para_cnt_freq[i - 1]
            para_cnt_freq[i] += para_cnt_freq[i - 1]
        plotAndSave(para_cnt_space, para_cnt_freq, "第" + str(convLayerCnt) + "卷积层卷积核参数分布CDF", "./fig_comp/cdf/" + str(convLayerCnt) + ".jpg")
        convLayerCnt += 1
        para_cnt_space = np.zeros(11)
        para_cnt_freq = np.zeros(11)
