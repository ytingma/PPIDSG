import torch.optim
import copy
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from src.EtC.EtC_cifar import *
from src.EtC.EtC_mnist import EtC_mnist
from PPIDSG.options import args_parser
from PPIDSG.utils import ImagePool


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


class LocalUpdate(object):
    def __init__(self, args, dataset, G, D_B, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = G
        self.D_B = D_B
        self.criterion1 = nn.MSELoss().to(self.device)
        self.criterion2 = nn.L1Loss().to(self.device)
        self.criterion3 = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        # split indexes for train
        idxs_train = idxs[:int(1*len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.batch_size, shuffle=True)
        return trainloader

    def update_weights(self, D_A_model, C, global_round):
        # Set mode to train model
        args = args_parser()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        D_A_losses = [] # feature extractor
        D_B_losses = [] # discriminator
        C_losses = [] # classifier
        G_losses = [] # generator

        # Set optimizer for the local updates
        G_optimizer = torch.optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        D_A_optimizer = torch.optim.SGD(D_A_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
        C_optimizer = torch.optim.SGD(C.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
        D_B_optimizer = torch.optim.Adam(self.D_B.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        # learning rate decay
        if (global_round + 1) > args.decay_epoch:
            D_A_optimizer.param_groups[0]['lr'] -= args.lr / (args.num_epochs - args.decay_epoch)
            C_optimizer.param_groups[0]['lr'] -= args.lr / (args.num_epochs - args.decay_epoch)
            D_B_optimizer.param_groups[0]['lr'] -= args.lrD / (args.num_epochs - args.decay_epoch)
            G_optimizer.param_groups[0]['lr'] -= args.lrG / (args.num_epochs - args.decay_epoch)

        # Generated image pool
        num_pool = 10
        fake_A_pool = ImagePool(num_pool)

        # local update
        for iter in range(1):
            for i, (real_A, label) in enumerate(self.trainloader):
                # input image data
                imgs = real_A.numpy().astype('float32')
                if args.dataset == 'mnist' or args.dataset == 'fmnist':
                    real_B = EtC_mnist(imgs)
                else:
                    real_B = EtC_cifar4(imgs)

                real_B = Variable(real_B.to(device))
                real_A = Variable(real_A.to(device))
                label = label.to(device)

                # Train generator G
                for j in range(args.epoch_G):
                    change_feature, fake_A = self.G(real_A)
                    D_B_fake_decision = self.D_B(fake_A)
                    G_feature_image = self.criterion1(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size()).to(device))) # Loss: adv
                    G_image_loss = self.criterion2(fake_A, real_B) # Loss: sem

                    fake_feature, fake_decoded = D_A_model(fake_A)
                    fake_predicts = C(fake_feature)
                    label_loss = self.criterion3(fake_predicts, label) * args.lambdaC # Loss: cls

                    # Back propagation
                    G_loss = G_feature_image + label_loss + G_image_loss
                    G_optimizer.zero_grad()
                    G_loss.backward(retain_graph=True)
                    G_optimizer.step()

                # Train feature extractor D_A
                change_feature, fake_A = self.G(real_A)
                fake_A1 = fake_A_pool.query(fake_A)
                fake_features, fake_decoded = D_A_model(fake_A1)
                D_A_fake_loss = self.criterion1(fake_decoded, fake_A1)

                # Back propagation
                D_A_loss = D_A_fake_loss
                D_A_optimizer.zero_grad()
                D_A_loss.backward(retain_graph=True)
                D_A_optimizer.step()

                # train classifier
                fake_features, _ = D_A_model(fake_A1)
                fake_predicts = C(fake_features)
                C_loss = self.criterion3(fake_predicts, label)

                # Back propagation
                C_optimizer.zero_grad()
                C_loss.backward(retain_graph=True)
                C_optimizer.step()

                # Train discriminator D_B
                D_B_real_decision = self.D_B(real_B)
                D_B_real_loss = self.criterion1(D_B_real_decision, Variable(torch.ones(D_B_real_decision.size()).to(device)))
                fake_A2 = fake_A_pool.query(fake_A)
                D_B_fake_decision = self.D_B(fake_A2)
                D_B_fake_loss = self.criterion1(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size()).to(device)))

                # Back propagation
                D_B_loss = (D_B_real_loss + D_B_fake_loss)
                D_B_optimizer.zero_grad()
                D_B_loss.backward(retain_graph=True)
                D_B_optimizer.step()

                # loss values
                D_A_losses.append(D_A_loss.item())
                D_B_losses.append(D_B_loss.item())
                G_losses.append(G_loss.item())
                C_losses.append(C_loss.item())

                '''if (i % 100 == 0):
                    print('Epoch [%d/%d], Step [%d/%d], D_A_loss: %.4f, D_B_loss: %.4f, G_loss: %.4f, C_loss: %.4f'
                          % (global_round + 1, args.num_epochs, i + 1, len(self.trainloader), D_A_loss.item(),
                             D_B_loss.item(), G_loss.item(), C_loss.item()))'''

        return D_A_model.state_dict(), self.G.state_dict(), self.D_B.state_dict(), C.state_dict()


def test_inference(G, D_A, C, test_dataset):
    """ Returns the test accuracy and loss.
    """
    device = 'cuda'
    G.to(device)
    C.to(device)
    loss, total, correct = 0.0, 0.0, 0.0
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    for batch_idx, (images, labels) in enumerate(testloader):
        real_A = images.to(device)
        labels = labels.to(device)

        # Inference
        _, fake_A = G(real_A)
        features, _ = D_A(fake_A)
        predicts = C(features)

        # Prediction
        _, pred_labels = torch.max(predicts, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy
