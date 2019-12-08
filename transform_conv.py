import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import cv2
import os

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator


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
    net = models.vgg16_bn()
    net.load_state_dict(torch.load('../vgg16_bn-6c64b313.pth'))
    net.to(device)
    freq_matrix = conv_dct(net)


def power_res(weight):
    """
    calculate the magnitude of parameter
    :param weight: parameter itself
    :return: magnitude of parameter, that is lg(abs(weight))+10
    """
    if weight < 0:
        weight = -weight
    if weight > 1:
        return 10
    elif weight < 1e-10:
        return 0
    else:
        res = math.floor(math.log(weight, 10)) + 10
    return res


def plot_save(w_space, w_freq, title, filename):
    """
    draw and save line chart
    :param w_space: parameter count in spatial domain
    :param w_freq: parameter count in frequency domain
    :param title: chart title
    :param filename: path for file saving
    :return: none
    """
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-10.5, 0.5)
    plt.ylim(-0.05, 1.05)
    plt.plot(np.arange(-10, 1), w_space, color="red", label='Parameters in spatial domain')
    plt.plot(np.arange(-10, 1), w_freq, color="skyblue", label='Parameters in frequency domain')
    plt.legend(loc='upper left')
    plt.ylabel('percentage')
    plt.xlabel('parament magnitude/lg')
    plt.title(title, FontProperties=font)
    plt.savefig(filename)
    plt.close()


conv_layer_cnt = 1
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
        print(conv_layer_cnt, '\t', f)
        filter_cnt, channel_cnt, kernel_width, kernel_weight = f.weight.size()[0], f.weight.size()[1], f.weight.size()[2], f.weight.size()[3]
        filter_size = channel_cnt * kernel_width * kernel_weight
        total_cnt = filter_cnt * filter_size
        filters_space = f.weight.data.view(filter_cnt, filter_size).cpu().numpy()
        filters_freq = freq_matrix[conv_layer_cnt - 1]
        for i in range(0, filter_cnt):
            for j in range(0, filter_size):
                cur_weight = filters_space[i][j]
                para_cnt_space[power_res(cur_weight)] += 1
                cur_weight = filters_freq[j][i]
                para_cnt_freq[power_res(cur_weight)] += 1
        para_cnt_space /= total_cnt
        para_cnt_freq /= total_cnt
        plot_save(para_cnt_space, para_cnt_freq, "第" + str(conv_layer_cnt) + "卷积层卷积核参数分布PDF", "./fig_comp/pdf/" + str(conv_layer_cnt) + ".jpg")
        for i in range(1, len(para_cnt_space)):
            para_cnt_space[i] += para_cnt_space[i - 1]
            para_cnt_freq[i] += para_cnt_freq[i - 1]
        plot_save(para_cnt_space, para_cnt_freq, "第" + str(conv_layer_cnt) + "卷积层卷积核参数分布CDF", "./fig_comp/cdf/" + str(conv_layer_cnt) + ".jpg")
        conv_layer_cnt += 1
        para_cnt_space = np.zeros(11)
        para_cnt_freq = np.zeros(11)
