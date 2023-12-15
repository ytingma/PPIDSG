import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PPIDSG.models import Generator

# hyperparameter setting
datas = 'FMNIST' # mnist, fmnist, cifar10, svhn
dummys = 'dummy_data' # dummy data: the random input, test:the image from test dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_pil_image = transforms.ToPILImage()

if datas == 'FMNIST' or datas == 'MNIST':
    target_model = Generator(1, 32, 1, 6).to(device)
else:
    target_model = Generator(3, 32, 3, 6).to(device)

# load the trained generater parameters from the client's sharing
if datas == 'CIFAR10':
    modelPath = './cifar_model/'
    target_model.load_state_dict(torch.load(modelPath + 'generator_param.pkl'))
elif datas == 'MNIST':
    modelPath = './mnist_model/'
    target_model.load_state_dict(torch.load(modelPath + 'generator_param.pkl'))
elif datas == 'SVHN':
    modelPath = './svhn_model/'
    target_model.load_state_dict(torch.load(modelPath + 'generator_param.pkl'))
elif datas == 'FMNIST':
    modelPath = './fmnist_model/'
    target_model.load_state_dict(torch.load(modelPath + 'generator_param.pkl'))

apply_transform = transforms.Compose([transforms.ToTensor()])

apply_transform_cifar = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

apply_transform_svhn = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

apply_transform_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])

if datas == 'SVHN':
    train_dataset = datasets.SVHN(root='./data/svhn/', split='train', download=False, transform=apply_transform_svhn)
    test_dataset = datasets.SVHN(root='./data/svhn/', split='test', download=False, transform=apply_transform_svhn)
elif datas == 'CIFAR10':
    train_dataset = datasets.CIFAR10(root='./data/cifar/', train=True, transform=apply_transform_cifar)
    test_dataset = datasets.CIFAR10(root='./data/cifar/', train=False, transform=apply_transform_cifar)
elif datas == 'MNIST':
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=apply_transform_mnist)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=apply_transform_mnist)
elif datas == 'FMNIST':
    train_dataset = datasets.FashionMNIST(root='./data/fmnist/', train=True, transform=apply_transform_mnist)
    test_dataset = datasets.FashionMNIST(root='./data/fmnist/', train=False, transform=apply_transform_mnist)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# image
test_loader_X = iter(test_loader).next()[0].clone().detach()
test_loader_X = test_loader_X.to(device)

# label
if datas == 'MNIST' or datas == 'FMNIST':
    test_loader_Y = test_loader_X[1].view(1, 1, 28, 28)
elif datas == 'CIFAR10':
    test_loader_Y = test_loader_X[1].view(1, 3, 32, 32)
elif datas == 'SVHN':
    test_loader_Y = test_loader_X[17].view(1, 3, 32, 32)

img = to_pil_image(test_loader_Y[0].cpu())
img.show()

# get the input of generator
if dummys == 'dummy_data':
    dummy_data = torch.randn(test_loader_Y.size()).to(device)
    img = to_pil_image(dummy_data[0].cpu())
    img.show()
else:
    dummy_data = test_loader_X
    img = to_pil_image(dummy_data[0].cpu())
    img.show()

# RS
_, pred_output = target_model(dummy_data)

# show reconstructed image
img = to_pil_image(pred_output[0].cpu())
img.show()
img.save('./picture/dummy_fmnist.jpg')
