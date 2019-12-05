import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchvision import transforms
from datetime import datetime


def myEvaluate(model, test_dataset, test_loader, criterion):
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()  # 模型评估
    eval_loss = 0
    eval_acc = 0
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
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()


def saveModel(model):
    torch.save(model.state_dict(), '../myVGG16_BN.pth')


def loadModel():
    resModel = myVGG(make_layers(vggStructure, batch_norm=True))
    resModel.load_state_dict(torch.load('../myVGG16_BN.pth'))
    return resModel


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().data.item()
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
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

            train_loss += loss.data.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda(), volatile=True)
                    label = Variable(label.cuda(), volatile=True)
                else:
                    im = Variable(im, volatile=True)
                    label = Variable(label, volatile=True)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.data.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)


vggStructure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class myVGG(nn.Module):
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

train_set = CIFAR10('../data', train=True, transform=data_tf, download=False)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10('../data', train=False, transform=data_tf, download=False)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

myNet = myVGG(make_layers(vggStructure, batch_norm=True))
learningRate = 0.1
optimizer = torch.optim.SGD(myNet.parameters(), learningRate, momentum=0.9, dampening=0, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

for i in range(0, 3):
    train(myNet, train_data, test_data, 100, optimizer, criterion)
    learningRate /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = learningRate

myEvaluate(myNet, test_set, test_data, criterion)
saveModel(myNet)

myNet = loadModel()
myEvaluate(myNet, test_set, test_data, criterion)