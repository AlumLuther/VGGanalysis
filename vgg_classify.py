import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchvision import transforms
from datetime import datetime


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().data.item()
    return num_correct / total


def evaluate(model, test_loader, criterion):
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()  # 模型评估
    test_loss = 0
    test_acc = 0
    for data in test_loader:  # 测试模型
        img, label = data
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        test_loss += loss.item() * label.size(0)
        test_acc += get_acc(out, label)
    resStr = 'Test Loss: {:.6f}, Acc: {:.6f}'.format(test_loss / (len(test_loader)), test_acc / (len(test_loader)))
    return resStr


def train(net, train_data, test_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train acc calc
            train_loss += loss.data.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        epoch_str = ("Epoch {%d}. ".format(epoch))
        train_str = ("Train Loss: {:.6f}, Train Acc: {:.6f}, ".format(train_loss / len(train_data), train_acc / len(train_data)))
        test_str = evaluate(net, test_data, criterion)
        print(epoch_str + train_str + test_str + time_str)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    resModel = MyVGG(make_layers(vgg_structure, batch_norm=True))
    resModel.load_state_dict(torch.load(path))
    return resModel


class MyVGG(nn.Module):
    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output


def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for curLayer in cfg:
        if curLayer == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, curLayer, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(curLayer)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = curLayer
    return nn.Sequential(*layers)


data_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_path = '../data'
model_path = '../MyVGG16_BN.pth'

train_set = CIFAR10(data_path, train=True, transform=data_tf, download=False)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10(data_path, train=False, transform=data_tf, download=False)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

vgg_structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
net = MyVGG(make_layers(vgg_structure, batch_norm=True))

learning_rate = 0.1
optimizer = torch.optim.SGD(net.parameters(), learning_rate, momentum=0.9, dampening=0, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# model training
for i in range(0, 3):
    train(net, train_data, test_data, 100, optimizer, criterion)
    learning_rate /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

evaluate(net, test_data, criterion)
save_model(net, model_path)

myNet = load_model(model_path)
evaluate(myNet, test_data, criterion)
