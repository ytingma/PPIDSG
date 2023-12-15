import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from src.EtC.EtC_cifar import EtC_cifar4
from src.EtC.EtC_mnist import EtC_mnist


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func1 = nn.MSELoss()
        self.ldr_train = self.train_val_test(dataset, list(idxs))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.idxs = idxs

    def train_val_test(self, dataset, idxs):
        # split indexes for train, validation, and test_alpha (100, 0, 0)
        idxs_train = idxs[:int(1*len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=64, shuffle=True)
        return trainloader

    def update_weights(self, net, global_round):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        if (global_round + 1) > self.args.decay_epoch:
            optimizer.param_groups[0]['lr'] -= self.args.lr / (self.args.epochs - self.args.decay_epoch)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)

                imgs = images.cpu().numpy().astype('float32')
                if(self.args.dataset=='fmnist' or self.args.dataset=='mnist'):
                    image = EtC_mnist(imgs).to(self.device) # mnist, fmnist: EtC_mnist; cifar10, svhn: EtC_cifar
                else:
                    image = EtC_cifar4(imgs).to(self.device)

                net.zero_grad()
                _, log_probs = net(image)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                optimizer.step() # update
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def test_img(net_g, datatest):
    net_g.eval()
    # testing
    loss = 0
    correct, total = 0.0, 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss().to(device)
    data_loader = DataLoader(datatest, batch_size=64)
    # test accuracy
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(device), target.cuda(device)
        imgs = data.cpu().numpy().astype('float32')
        image = EtC_mnist(imgs).to(device)
        _, outputs = net_g(image)

        batch_loss = criterion(outputs, target)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, target)).item()
        total += len(target)

    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy

