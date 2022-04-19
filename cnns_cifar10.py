# encoding=utf-8

# from:
# https://github.com/kuangliu/pytorch-cifar

import os
import sys
import platform
import torch
import torchvision
from torchvision import transforms, datasets
from torchsummary import summary
from torch import nn
from torch.nn import functional as F
import torch.utils.data  # 如果不用这个就会出现pycharm不识别data的问题
import math
from models.lenet import LeNet
from models.vgg import VGG
from models.resnet import BasicBlock, Bottleneck, ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.resnext import ResNeXt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



if __name__ == '__main__':
    # net = LeNet()
    # net = VGG('VGG11')
    # net = VGG('VGG13')
    # net = VGG('VGG16')
    # net = VGG('VGG19')
    # net = ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18
    # net = ResNet(BasicBlock, [3, 4, 6, 3])  # ResNet34
    # net = ResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
    # net = ResNet(Bottleneck, [3, 4, 23, 3])  # ResNet101
    # net = ResNet(Bottleneck, [3, 8, 36, 3])  # ResNet152
    # net = ResNet152() # ResNet152
    # net = ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64) # ResNeXt29_2x64d
    # net = ResNeXt(num_blocks=[3, 3, 3], cardinality=4, bottleneck_width=64)   # ResNeXt29_4x64d
    # net = ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64) # ResNeXt29_8x64d
    # net = ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4)   # ResNeXt29_32x4d
    # net = GoogLeNet()
    # net = DenseNet(Bottleneck_DenseNet, [6, 12, 24, 16], growth_rate=32) # DenseNet121
    # net = DenseNet(Bottleneck_DenseNet, [6,12,32,32], growth_rate=32) # DenseNet169
    # net = DenseNet(Bottleneck_DenseNet, [6,12,48,32], growth_rate=32) # DenseNet201
    # net = DenseNet(Bottleneck_DenseNet, [6,12,36,24], growth_rate=48) # DenseNet161
    # net = DenseNet(Bottleneck_DenseNet, [6,12,24,16], growth_rate=12) # densenet_cifar
    # print(net)
    # summary(net, (3, 32, 32))
    # exit(0)

    print(platform.python_version())
    print(platform.python_version_tuple())
    print(type(platform.python_version_tuple()))

    print('python version')
    print(sys.version)
    print('version info')
    print(sys.version_info)
    print('pytorch version')
    print(torch.__version__)

    # 3.7.5
    # ('3', '7', '5')
    # <class 'tuple'>
    # python version
    # 3.7.5 (v3.7.5:5c02a39a0b, Oct 14 2019, 18:49:57)
    # [Clang 6.0 (clang-600.0.57)]
    # version info
    # sys.version_info(major=3, minor=7, micro=5, releaselevel='final', serial=0)
    # pytorch version
    # 1.11.0

    # # cifar10
    # transform_train = transforms.Compose(
    #     [
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ]
    # )
    # transform_test = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ]
    # )
    # trainset = torchvision.datasets.CIFAR10('./data', train=True, transform=transform_train, download=True)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
    # testset = torchvision.datasets.CIFAR10('./data', train=False, transform=transform_test, download=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=0)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # hymenoptera
    # from:
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # https://fancyerii.github.io/books/pytorch/
    # data:
    # https://download.pytorch.org/tutorial/hymenoptera_data.zip
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = './data/hymenoptera_data'
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in ['train', 'val']
    }
    dataset_size = {
        x: len(image_datasets[x]) for x in ['train', 'val']
    }
    class_names = image_datasets['train'].classes

    net = LeNet()
    # net = VGG('VGG11')
    # ...
    print(net)
    summary(net, (3, 32, 32))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    learning_rate = 0.1
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    resume = False
    if resume:
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # test
    epoch = 10
    # epoch = 200

    for epoch in range(start_epoch, start_epoch + epoch):
        # trainset
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batch = 0
        # for batch_idx, (inputs, targets) in enumerate(trainloader):
        for batch_idx, (inputs, targets) in enumerate(dataloaders['train']):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            train_batch = batch_idx + 1

            # # test
            # if batch_idx >= 2:
            #     break

        # testset
        net.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        test_batch = 0
        # for batch_idx, (inputs, targets) in enumerate(testloader):
        for batch_idx, (inputs, targets) in enumerate(dataloaders['val']):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            test_batch = batch_idx + 1

            # # test
            # if batch_idx >= 2:
            #     break

        print('epoch: {}/{}, train loss={:.4f}, train acc={:.2f}%, test loss={:.4f}, test acc={:.2f}%'.format(
            epoch + 1, start_epoch + 200,
            train_loss / train_batch, train_correct * 100.0 / train_total,
            test_loss / test_batch, test_correct * 100.0 / test_total
        ))

        # save best acc
        test_acc = test_correct * 1.0 / test_total
        if test_acc > best_acc:
            print('saving...')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = test_acc

        scheduler.step()
