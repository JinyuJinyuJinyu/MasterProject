import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
resnet18 = models.resnet18()

class Identity(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1,):
        super(Identity, self).__init__()

        self.con2a = nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2a = nn.BatchNorm2d(planes)

        self.con2b = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2b = nn.BatchNorm2d(planes)
        if stride !=1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = lambda x: x
    def forward(self, input_tensor):
        x = self.con2a(input_tensor)
        x = self.bn2a(x)
        x = F.relu(x)

        x = self.con2b(x)
        x = self.bn2(x)

        identity = self.downsample(x)
        x += identity
        return F.relu(x)


class Resnet_s(nn.Module):
    def __init__(self,block,layers,classes=10):
        super(Resnet_s,self).__init__()
        self.inplanes = 64

        self.conv2a = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2a = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,classes)

    def _make_layer(self,block,planes,blocks,stride=1):

        layers = []
        for _ in range(1, blocks):

            layers.append(block(self.inplanes,planes,stride=stride))

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



model = Resnet_s(Identity, [2, 2, 2, 2])
print(model)