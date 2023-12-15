import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import copy
import torch
import time
import os

from tqdm import tqdm
from PPIDSG.options import args_parser
import torch.nn.functional as F
from PPIDSG.update import LocalUpdate, test_inference
from PPIDSG.models import Generator, Discriminator, AutoEncoder_VGG, VGG16_classifier, AutoEncoder_VGG_mnist, VGG16_classifier_mnist
from PPIDSG.utils import get_dataset, average_weights
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report
from src.MIA.utils import train_attack_model
from src.models import AttackMLP
np.random.seed(1234)

# Attack Model Hyper-parameters
NUM_EPOCHS = 100
BATCH_SIZE = 128
num_workers = 0
args = args_parser()


# print MIA details
def attack_inference(model, test_X, test_Y, device):
    print('----Attack Model Testing----')

    targetnames = ['non-member', 'Member']
    pred_y = []
    true_y = []
    X = torch.cat(test_X)
    Y = torch.cat(test_Y)

    inferset = TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset=inferset, batch_size=50, shuffle=False, num_workers=0)

    # Evaluation of Attack Model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            true_y.append(labels.cpu())
            pred_y.append(predictions.cpu())

    attack_acc = correct / total
    print('Attack Test Accuracy is  : {:.2f}%'.format(100 * attack_acc))

    true_y = torch.cat(true_y).numpy()
    pred_y = torch.cat(pred_y).numpy()

    print('---Detailed Results----')
    print(classification_report(true_y, pred_y, target_names=targetnames))


# create shadow dataset
def split_shadow_dataset(test_dataset):
    total_size = len(test_dataset)
    split1 = total_size // 2

    indices = list(range(total_size))
    np.random.shuffle(indices)

    # Shadow model train and test_alpha set
    s_train_idx = indices[:split1]
    s_test_idx = indices[split1:]

    return s_train_idx, s_test_idx


# the core of MIA (overfitting)
def prepare_target_data(G, Fea, C, iterator, device, victim):
    attackX = []
    attackY = []

    with torch.no_grad():
        for inputs, _ in iterator:
            inputs = inputs.to(device)
            _, fake_image = G(inputs)
            features, _ = Fea(fake_image)
            output = C(features)

            posteriors = F.softmax(output, dim=1)
            attackX.append(posteriors.cpu())

            if victim:
                attackY.append(torch.ones(posteriors.size(0), dtype=torch.long))
            else:
                attackY.append(torch.zeros(posteriors.size(0), dtype=torch.long))
    return attackX, attackY


def get_data_loader(dataset='mnist', batch=64, shadow_split=0.5, num_workers=0):
    dataset = args.dataset # mnist, fmnist, cifar10, svhn
    if dataset == 'mnist':
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset == 'fmnist':
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
    else:
        test_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    if dataset == 'mnist':
        test_set = torchvision.datasets.MNIST(root='./data/', train=False, download=False,
                                              transform=test_transforms)
    elif dataset == 'cifar':
        test_set = torchvision.datasets.CIFAR10(root='./data/cifar/', train=False, download=False,
                                                transform=test_transforms)
    elif dataset == 'svhn':
        test_set = torchvision.datasets.SVHN(root='./data/svhn/', split='test', download=False,
                                             transform=test_transforms)
    elif dataset == 'fmnist':
        test_set = torchvision.datasets.FashionMNIST(root='./data/fmnist/', train=False, download=False,
                                                     transform=test_transforms)

    s_train_idx, s_out_idx = split_shadow_dataset(test_set)

    # Data samplers
    s_train_sampler = SubsetRandomSampler(s_train_idx)
    s_out_sampler = SubsetRandomSampler(s_out_idx)

    s_train_loader = DataLoader(dataset=test_set, batch_size=batch, sampler=s_train_sampler, num_workers=num_workers)
    s_out_loader = DataLoader(dataset=test_set, batch_size=batch, sampler=s_out_sampler, num_workers=num_workers)

    print('Total Test samples in {} dataset : {}'.format(dataset, len(test_set)))
    print('Number of Shadow train samples: {}'.format(len(s_train_sampler)))
    print('Number of Shadow test_alpha samples: {}'.format(len(s_out_sampler)))

    return s_train_loader, s_out_loader


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


# MIA
def create_attack(num, dataset, victim, others):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Creating data loaders
    s_train_loader, s_test_loader = get_data_loader()

    idxs_train = victim[:int(1 * len(victim))]
    t_train_loader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=64, shuffle=False)
    idxs_test = others[:int(1 * len(others))]
    t_test_loader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=64, shuffle=False)

    # target madel
    if args.dataset=='mnist':
        G = Generator(1, 32, 1, 6).to(device)  # cifar10, svhn: channel(3); mnist, fmnist: channel(1)
        G_O = Generator(1, 32, 1, 6).to(device)  # cifar10, svhn: channel(3); mnist, fmnist: channel(1)
        C = VGG16_classifier_mnist().to(device)
        Fea = AutoEncoder_VGG_mnist().to(device)  # cifar10, svhn: AutoEncoder_VGG; mnist, fmnist: AutoEncoder_VGG_mnist
        F_O = AutoEncoder_VGG_mnist().to(device)  # cifar10, svhn: AutoEncoder_VGG; mnist, fmnist: AutoEncoder_VGG_mnist
    elif args.dataset == 'fmnist':
        G = Generator(1, 32, 1, 6).to(device)
        G_O = Generator(1, 32, 1, 6).to(device)
        C = VGG16_classifier_mnist().to(device)
        Fea = AutoEncoder_VGG_mnist().to(device)
        F_O = AutoEncoder_VGG_mnist().to(device)
    else:
        G = Generator(3, 32, 3, 6).to(device)
        G_O = Generator(3, 32, 3, 6).to(device)
        C = VGG16_classifier().to(device)
        Fea = AutoEncoder_VGG().to(device)
        F_O = AutoEncoder_VGG().to(device)

    modelPath = './cifar_model/' # the model dir
    G.load_state_dict(torch.load(modelPath + '0_generator_param.pkl')) # victim generator
    C.load_state_dict(torch.load(modelPath + '_classifier_param.pkl')) # classifier
    Fea.load_state_dict(torch.load(modelPath + '0_extractor_param.pkl')) # victim feature extractor
    if num == 1:
        G_O.load_state_dict(torch.load(modelPath + '1_generator_param.pkl')) # other generator (part situation)
        F_O.load_state_dict(torch.load(modelPath + '1_extractor_param.pkl')) # other feature extractor
    else:
        G_O.load_state_dict(torch.load(modelPath + 'others_generator_param.pkl')) # others generator
        F_O.load_state_dict(torch.load(modelPath + 'others_extractor_param.pkl')) # others feature extractor (all situation)

    # the skeptical dataset
    t_trainX, t_trainY = prepare_target_data(G, Fea, C, t_train_loader, device, victim=True)
    t_testX, t_testY = prepare_target_data(G_O, F_O, C, t_test_loader, device, victim=False)
    targetX = t_trainX + t_testX
    targetY = t_trainY + t_testY

    print("shadow model") # the shadow dataset
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        F_ini = AutoEncoder_VGG_mnist().to(device)  # attacker cannot get the feature extractor
    else:
        F_ini = AutoEncoder_VGG().to(device)
    print('----Preparing Attack training data---')
    trainX, trainY = prepare_target_data(G, F_ini, C, s_train_loader, device, victim=True)
    testX, testY = prepare_target_data(G_O, F_ini, C, s_test_loader, device, victim=False)
    shadowX = trainX + testX
    shadowY = trainY + testY

    # Attack Model Training
    input_size = shadowX[0].size(1)
    n_hidden = 128
    print('Input Feature dim for Attack Model')

    attack_model = AttackMLP(input_size=input_size, hidden_size=n_hidden, out_classes=2).to(device)

    # Loss and optimizer
    attack_loss = nn.CrossEntropyLoss()
    attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.005, weight_decay=1e-4)
    attack_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attack_optimizer, gamma=0.96)

    # Feature vector and labels for training Attack model
    attackdataset = (shadowX, shadowY)
    modelDir = os.path.join(modelPath)
    kptpath = 'best_attack_model.ckpt'

    attack_valacc = train_attack_model(attack_model, attackdataset, attack_loss,
                                       attack_optimizer, attack_lr_scheduler, device, modelDir,
                                       kptpath, NUM_EPOCHS, BATCH_SIZE, num_workers)

    print('Validation Accuracy for the Best Attack Model is: {:.2f} %'.format(100 * attack_valacc))

    # Load the trained attack model
    attack_path = os.path.join(modelDir, 'best_attack_model.ckpt')
    attack_model.load_state_dict(torch.load(attack_path))
    attack_inference(attack_model, targetX, targetY, device) # attack testing


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    model_dir ='./cifar_model/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.dataset=='mnist':
        G = Generator(1, args.ngf, 1, args.num_resnet)
        D_B = Discriminator(1, args.ndf, 1)
        global_model = AutoEncoder_VGG_mnist().to(device)
        C = VGG16_classifier_mnist().to(device)
    elif args.dataset == 'fmnist':
        G = Generator(1, args.ngf, 1, args.num_resnet)
        D_B = Discriminator(1, args.ndf, 1)
        global_model = AutoEncoder_VGG_mnist().to(device)
        C = VGG16_classifier_mnist().to(device)
    else:
        G = Generator(3, args.ngf, 3, args.num_resnet)
        D_B = Discriminator(3, args.ndf, 1)
        global_model = AutoEncoder_VGG().to(device)
        C = VGG16_classifier().to(device)

    G.normal_weight_init(mean=0.0, std=0.02)
    D_B.normal_weight_init(mean=0.0, std=0.02)
    G.to(device)
    D_B.to(device)

    # copy weights
    global_weights = global_model.state_dict()
    G_weights = G.state_dict()
    D_B_weights = D_B.state_dict()
    C_weights = C.state_dict()
    torch.save(C_weights, model_dir + '_classifier_param.pkl') # classifier key
    torch.save(global_weights, model_dir + '_extractor_param.pkl') # feature extractor

    # Training
    global_model_local_weights = [global_weights for i in range(args.num_users)]
    D_B_local_weights = [D_B_weights for i in range(args.num_users)]
    C_local_weights = [C_weights for i in range(args.num_users)]

    for epoch in tqdm(range(args.num_epochs)):
        G_local_weights = []
        sum_weights1, sum_weights2, sum_weights3 = [], [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        for idx in range(args.num_users):
            global_model.load_state_dict(global_model_local_weights[idx])
            D_B.load_state_dict(D_B_local_weights[idx])
            C.load_state_dict(C_local_weights[idx])

            local_model = LocalUpdate(args=args, dataset=train_dataset, G=copy.deepcopy(G),
                                      D_B=copy.deepcopy(D_B), idxs=user_groups[idx])
            w, v, u, z = local_model.update_weights(D_A_model=copy.deepcopy(global_model),
                                                    C=copy.deepcopy(C), global_round=epoch)
            G_local_weights.append(copy.deepcopy(v))
            D_B_local_weights[idx] = copy.deepcopy(u)
            global_model_local_weights[idx] = copy.deepcopy(w)

            if epoch > 48:
                torch.save(v, model_dir + str(idx) + '_generator_param.pkl') # victim generator
                torch.save(w, model_dir + str(idx) + '_extractor_param.pkl') # victim feature extractor
                torch.save(z, model_dir + str(idx) + '_classifier_param.pkl') # victim classifier
                if idx != 0:
                    sum_weights1.append(copy.deepcopy(v)) # client generator
                    sum_weights2.append(copy.deepcopy(w)) # client feature extractor
                    sum_weights3.append(copy.deepcopy(z)) # client classifier

        # update global weights and local weights
        G_weights = average_weights(G_local_weights)
        G.load_state_dict(G_weights)

        if epoch > 48:
            torch.save(G_weights, model_dir + 'generator_param.pkl')

        test_acc = test_inference(G, global_model, C, test_dataset) # test accuracy
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    weights = average_weights(sum_weights1) # others generator (all situation)
    torch.save(weights, model_dir + 'others_generator_param.pkl')

    weights = average_weights(sum_weights2) # others feature extractor (all situation)
    torch.save(weights, model_dir + 'others_extractor_param.pkl')

    others = []
    # get others data index (all situation)
    for i in range(9):
        others = others + list(user_groups[i + 1])

    # MIA
    print('------num=1---------') # part situation
    create_attack(num=1, dataset=train_dataset, victim=list(user_groups[0]), others=list(user_groups[1]))
    print('------num=10---------') # all situation
    create_attack(num=10, dataset=train_dataset, victim=list(user_groups[0]), others=others)
