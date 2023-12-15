import copy
import torch
import random
import glob
import os
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)


def dataset_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(1234)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def dataset_split(train_dataset, num_users):
    np.random.seed(1234)
    total_size = len(train_dataset)
    split1 = total_size // num_users

    indices = list(range(total_size))
    np.random.shuffle(indices)

    dict_users = {}
    # Target model train and test_alpha set
    t_train_idx = indices[:split1]
    t_test_idx = indices[split1:]

    dict_users[0] = t_train_idx
    dict_users[1] = t_test_idx
    return dict_users


def get_dataset(args):
    if args.dataset == 'cifar':
        data_dir = './data/cifar/'

        apply_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=apply_transform)

        # sample training data amongst users
        user_groups = dataset_iid(train_dataset, args.num_users)

    elif args.dataset == 'svhn':
        data_dir = './data/svhn/'

        apply_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.SVHN(data_dir, split='train', download=False, transform=apply_transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=False, transform=apply_transform)

        # sample training data amongst users
        user_groups = dataset_iid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        if args.dataset == 'mnist':
            data_dir = './data/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=apply_transform)
        else:
            data_dir = './data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=False, transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=False, transform=apply_transform)

        # sample training data amongst users
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


def average_weights_new(w, p):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w[1][key], (1/p)) + torch.mul(w_avg[key], (1-(1/p)))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    return


# load historical images
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
