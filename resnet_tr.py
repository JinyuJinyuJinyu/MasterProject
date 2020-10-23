import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl

import torch_utils
import numpy as np
from sklearn.metrics import confusion_matrix
import time
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



def main():

    outfile = []
    f = open('SGD_tr.json',"w", encoding='utf-8')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = Resnet_s(Identity, [2, 2, 2, 2]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    val_time = 0
    print('start training Pytorch')
    start_time = time.time()
    for epoch in range(1,201):

        model.train(True)
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

        if epoch % 4 == 0:

            val_start_time = time.time()
            print('validating','epoch: ',epoch)
            confusion_mtx = torch.zeros((10,10)).cuda()
            val_info = {}
            losses = 0
            model.train(False)
            with torch.no_grad():
                for i, data in enumerate(testloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    prob = F.softmax(outputs, dim=1)
                    preds = torch.argmax(prob, dim=1)

                    # cm = confusion_matrix(labels.numpy(),preds.numpy())
                    cm = torch_utils.confusion_matrix(preds,labels,num_classes=10)
                    confusion_mtx = torch.add(confusion_mtx,cm)

                    # corrects += np.trace(cm)
                    losses += loss

            val_info['epoch'] = epoch
            val_info['test loss'] = losses.cpu().numpy().tolist()
            val_info['test accu'] = torch.trace(confusion_mtx).cpu().numpy()/100
            val_info['confusion matrix'] = confusion_mtx.tolist()
            outfile.append(val_info)
            val_time += time.time() - val_start_time

    ttl_time = {}
    ttl_time['training time'] = (time.time() - start_time - val_time)
    ttl_time['total time'] = (time.time() - start_time)
    ttl_time['val time'] = val_time
    outfile.append(ttl_time)
    json.dump(outfile, f, separators=(',', ':'), indent=4)
    f.close()

if __name__ == '__main__':
    main()
