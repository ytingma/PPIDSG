import copy
import torch
import numpy as np
from torchvision import datasets, transforms


def dataset_iid(dataset, num_users):
    np.random.seed(1234)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def get_dataset(args):

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=apply_transform)

        user_groups = dataset_iid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':
        if args.dataset == 'mnist':
            data_dir = './data/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=apply_transform)

        user_groups = dataset_iid(train_dataset, args.num_users)

    elif args.dataset == 'fmnist':
        data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=False, transform=apply_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=False, transform=apply_transform)

        user_groups = dataset_iid(train_dataset, args.num_users)

    elif args.dataset == 'svhn':
        data_dir = './data/svhn/'
        print("svhn")

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.SVHN(data_dir, split='train', download=False, transform=apply_transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=False, transform=apply_transform)

        user_groups = dataset_iid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    return
