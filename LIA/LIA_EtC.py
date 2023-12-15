import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from src.EtC.EtC_cifar import EtC_cifar4
from src.EtC.EtC_mnist import EtC_mnist
from src.models import LeNetZhu, ConvNet, ResNet18, LeNetZhu_mnist


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())


def main():
    dataset = 'FMNIST'
    # add your dataset position
    if dataset == 'MNIST':
        data_path = os.path.join('mnist_path')
    elif dataset == 'CIFAR10':
        data_path = os.path.join('cifar10_path')
    elif dataset == 'SVHN':
        data_path = os.path.join('svhn_path')
    elif dataset == 'FMNIST':
        data_path = os.path.join('fmnist_path')
    else:
        print("false dataset")

    num_dummy = 64 # batch size
    epoch = 10
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    tt = transforms.Compose([
        # transforms.Resize(32), # mnist, fmnist when they need 32*32 not 28*28
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)), # mnist, fmnist
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # svhn, cifar10
        ])

    if dataset == 'MNIST':
        dst = datasets.MNIST(data_path, download=False)
    elif dataset == 'CIFAR10':
        dst = datasets.CIFAR10(data_path, download=False)
    elif dataset == 'SVHN':
        dst = datasets.SVHN(data_path, download=False)
    elif dataset == 'FMNIST':
        dst = datasets.FashionMNIST(data_path, download=False)
    else:
        print("false dataset")

    for i in range(epoch):
        if dataset == 'MNIST':
            C = ConvNet(width=8, num_channels=1, num_classes=10).to(device)  # change the channel if you need
            # C = ResNet18(1).to(device) # change the input channel to 1
            # C = LeNetZhu_mnist().to(device)
        elif dataset=='FMNIST':
            C = ConvNet(width=8, num_channels=1, num_classes=10).to(device)  # change the channel if you need
            # C = ResNet18(1).to(device) # change the input channel to 1
            # C = LeNetZhu_mnist().to(device)
        else:
            C = ConvNet(width=8, num_channels=3, num_classes=10).to(device)  # change the channel if you need
            # C = ResNet18(3).to(device)
            # C = LeNetZhu().to(device)
        CEloss = nn.CrossEntropyLoss().to(device)
        imidx_list = [1098, 45985, 13093, 18025, 1896, 7997, 21102, 2051, 8661, 2599, 23255, 37992, 23310, 17812,
                      26366, 26611, 41645, 4451, 47841, 20541, 5041, 40320, 23289, 46633, 27264, 49472, 433, 1873,
                      46418, 41071, 48775, 14891, 37472, 15549, 45471, 32518, 35271, 10529, 11003, 19439, 1040, 32469,
                      17115, 13025, 24256, 32127, 2862, 36235, 45046, 13603, 18836, 1718, 23668, 29205, 37331, 49824,
                      22224, 15294, 30473, 16698, 16865, 25439, 36651, 15204] # an example
        for imidx in range(num_dummy):
            idx = imidx_list[imidx]
            tmp_datum = tt(dst[idx][0]).float().to(device)

            tmp_datum = tmp_datum.view(1, *tmp_datum.size())
            tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
            tmp_label = tmp_label.view(1, )
            if imidx == 0:
                gt_data = tmp_datum
                gt_label = tmp_label
            else:
                gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                gt_label = torch.cat((gt_label, tmp_label), dim=0)

        # print('----------original label-----------')
        # print(gt_label)

        imgs = gt_data.cpu().numpy().astype('float32')
        if dataset == 'MNIST' or dataset == 'FMNIST':
            images = EtC_mnist(imgs).to(device)  # the core of EtC, there are four ways(32*32)/two ways(28*28) to choose
        else:
            images = EtC_cifar4(imgs).to(device) # the core of EtC, there are four ways(32*32)/two ways(28*28) to choose
        _, predicts = C(images)
        loss = CEloss(predicts, gt_label)
        dy_dx = torch.autograd.grad(loss, C.parameters()) # get gradients
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
        tmp_min = torch.sum(original_dy_dx[-2], dim=-1)
        dummy_o, dummy_z = C(dummy_data)
        dummy_z, dummy_o = dummy_z.detach().clone(), dummy_o.detach().clone()
        dummy_p = F.softmax(dummy_z, dim=-1)
        sum_p = torch.sum(dummy_p, dim=0)
        sum_o = torch.sum(torch.mean(dummy_o, dim=0))
        a = (num_dummy / sum_o).to(device)
        b = tmp_min * a
        label_p = (sum_p - b).detach().clone().cpu().numpy()
        reds = np.round(label_p) # get labels
        '''print('--------------------------------------------------------------------------')'''
        print('num_label:', reds)


if __name__ == '__main__':
    main()
