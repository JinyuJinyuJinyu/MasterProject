import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import json



class Identity(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1,):
        super(Identity, self).__init__()

        self.con2a = nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2a = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.con2b = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2b = nn.BatchNorm2d(planes)
        if stride !=1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None
    def forward(self, input_tensor):
        identity = input_tensor
        x = self.con2a(input_tensor)
        x = self.bn2a(x)
        x = self.relu(x)

        x = self.con2b(x)
        x = self.bn2b(x)

        if self.downsample != None:
            identity = self.downsample(identity)
        x += identity
        return F.relu(x)


class Resnet_s(nn.Module):
    def __init__(self,block,layers,classes=10):
        super(Resnet_s,self).__init__()
        self.inplanes = 64

        self.conv2a = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2a = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,classes)

    def _make_layer(self,block,planes,blocks,stride=1):

        layers = []

        strides = [stride] + [1] * (blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:

            layers.append(block(self.inplanes,planes,stride=stride))
            self.inplanes = planes

        return nn.Sequential(*layers)



    def forward(self,input):
        x = self.conv2a(input)
        x = self.bn2a(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



# model = Resnet_s(Identity, [2, 2, 2, 2])
# print(model)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Hyper parameters
num_epochs = 2
num_classes = 10
batch_size = 64
learning_rate = 1e-3


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Resnet_s(Identity, [2, 2, 2, 2]).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1,2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device),data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


if __name__ == '__main__':
    main()
