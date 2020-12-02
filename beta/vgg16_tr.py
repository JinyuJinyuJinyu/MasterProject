import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader

import torch_utils
import utils
import time
import json

import warnings

warnings.filterwarnings("ignore", message="You have set 1000 number of classes which is different")

torch.manual_seed(0)

class Block1(nn.Module):

    def __init__(self,dim, stride=1):
        super(Block1,self).__init__()
        inplanes = dim[0]
        planes = dim[1]
        self.con2a = nn.Conv2d(inplanes,planes,kernel_size=(3, 3),stride=stride,padding=(1, 1))
        self.bn2a = nn.BatchNorm2d(planes)
        self.relua = nn.ReLU(inplace=True)


        self.con2b = nn.Conv2d(planes,planes,kernel_size=(3, 3),stride=stride,padding=(1, 1))
        self.bn2b = nn.BatchNorm2d(planes)
        self.relub = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self,input_tensor):
        x = self.con2a(input_tensor)
        x = self.bn2a(x)
        x = self.relua(x)

        x = self.con2b(x)
        x = self.bn2b(x)
        x = self.relub(x)

        x = self.pool(x)

        return x

class Block2(nn.Module):

    def __init__(self,dim, stride=1):
        super(Block2,self).__init__()
        inplanes = dim[0]
        planes = dim[1]
        self.con2a = nn.Conv2d(inplanes,planes,kernel_size=(3, 3),stride=stride,padding=(1, 1))
        self.bn2a = nn.BatchNorm2d(planes)
        self.relua = nn.ReLU(inplace=True)


        self.con2b = nn.Conv2d(planes,planes,kernel_size=(3, 3),stride=stride,padding=(1, 1))
        self.bn2b = nn.BatchNorm2d(planes)
        self.relub = nn.ReLU(inplace=True)

        self.con2c = nn.Conv2d(planes, planes,kernel_size=(3, 3), stride=stride,padding=(1, 1))
        self.bn2c = nn.BatchNorm2d(planes)
        self.reluc = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self,input_tensor):
        x = self.con2a(input_tensor)
        x = self.bn2a(x)
        x = self.relua(x)

        x = self.con2b(x)
        x = self.bn2b(x)
        x = self.relub(x)

        x = self.con2c(x)
        x = self.bn2c(x)
        x = self.reluc(x)

        x = self.pool(x)

        return x

class VGG16(nn.Module):

    def __init__(self,num_classes):
        super(VGG16,self).__init__()

        self.block2x = self._make_layers(Block1,[[3,64],[64,128]])
        self.block3x = self._make_layers(Block2,[[128,256],[256,512],[512,512]])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.densa = nn.Linear(25088,4096)
        self.relua = nn.ReLU(inplace=True)
        self.dpota = nn.Dropout(p=0.5, inplace=False)

        self.densb = nn.Linear(4096,4096)
        self.relub = nn.ReLU(inplace=True)
        self.dpotb = nn.Dropout(p=0.5, inplace=False)

        self.densc = nn.Linear(4096,num_classes)

    # build block layers
    def _make_layers(self, block, dims,stride=1):
        layers = []
        for dim in dims:

            layers.append(block(dim,stride=stride))
        return nn.Sequential(*layers)

    def forward(self,input_tensor):
        x = self.block2x(input_tensor)
        x = self.block3x(x)

        x = self.avgpool(x)

        x = torch.flatten(x,1)
        x = self.densa(x)
        x = self.relua(x)
        x = self.dpota(x)

        x = self.densb(x)
        x = self.relub(x)
        x = self.dpotb(x)

        x = self.densc(x)
        return x

batch_size = 32
epochs = 200

decayRate = 0.96



# torch.float32 torch.int64
# x_train, x_test, y_train, y_test = utils.load_dat()
# # use it later in calculate accuracy and loss
# number_val_samples = x_test.shape[0]
#
# # load data and make data type right, and reshape data to (N,W,H,C).(number of samples, weight, height, channel).
# # TF is (N,C,W,H)
# trains = torch.tensor(x_train)
# trains_label = torch.IntTensor(y_train)
# trains = torch.reshape(trains,(trains.shape[0],trains.shape[3],trains.shape[1],trains.shape[2]))
#
# tests = torch.tensor(x_test)
# tests_label = torch.IntTensor(y_test)
# tests = torch.reshape(tests,(tests.shape[0],tests.shape[3],tests.shape[1],tests.shape[2]))
#
# trains = trains.type(torch.float32)
# trains_label = trains_label.type(torch.int64)
# tests = tests.type(torch.float32)
# tests_label = tests_label.type(torch.int64)
#
# # print(trains.dtype,trains_label.dtype,tests.dtype,tests_label.dtype)
# trainset = TensorDataset(trains, trains_label)
# trainset_loader = DataLoader(trainset, batch_size,num_workers=2)
#
# testset = TensorDataset(tests,tests_label)
# testset_loader = DataLoader(testset, batch_size, num_workers=2)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testset_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

number_val_samples = 10000
def main(idx):

    outfile = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg = VGG16(10).to(device)

    optis = [optim.SGD(vgg.parameters(), lr=0.1, momentum=0.9)]
    f_name = ['SGD_vgg16_tr.json']

    fname = f_name[idx]
    optimizer = optis[idx]

    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)


    f = open(fname, "w", encoding='utf-8')

    criterion = nn.CrossEntropyLoss()
    val_time = 0
    print('start training Pytorch')
    start_time = init_time = time.time()
    for epoch in range(1, epochs + 1):

        vgg.train(True)
        for i, data in enumerate(trainset_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = vgg(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            my_lr_scheduler.step()

        if True:

            val_start_time = time.time()
            print('validating', 'epoch: ', epoch)
            confusion_mtx = torch.zeros((10, 10)).cuda()
            val_info = {}
            losses = 0
            mini_batch_count = 0
            vgg.train(False)
            with torch.no_grad():
                for i, data in enumerate(testset_loader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = vgg(inputs)

                    loss = criterion(outputs, labels)
                    prob = F.softmax(outputs, dim=1)
                    preds = torch.argmax(prob, dim=1)

                    cm = torch_utils.confusion_matrix(preds, labels, num_classes=10)
                    confusion_mtx = torch.add(confusion_mtx, cm)

                    # corrects += np.trace(cm)
                    losses += loss
                    mini_batch_count += 1

            val_info['epoch'] = epoch
            val_info['loss'] = losses.cpu().numpy().tolist() / mini_batch_count
            val_info['accu'] = torch.trace(confusion_mtx).cpu().numpy() / number_val_samples * 100
            print('epoch:  ', val_info['epoch'], '  loss:  ', val_info['loss'], '  accu: ', val_info['accu'])
            if epoch % epochs == 0:
                val_info['confusion matrix'] = confusion_mtx.tolist()
            outfile.append(val_info)
            val_time += time.time() - val_start_time
        if epoch == 1:
            init_time = time.time() - init_time


    ttl_time = {}
    ttl_time['training time'] = (time.time() - start_time - val_time)
    ttl_time['total time'] = (time.time() - start_time)
    ttl_time['val time'] = val_time
    ttl_time['init time'] = init_time
    outfile.append(ttl_time)
    json.dump(outfile, f, separators=(',', ':'), indent=4)
    f.close()

if __name__ == '__main__':
    for i in range(2):
        main(i)