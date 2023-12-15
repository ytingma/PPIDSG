import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=False, default='mnist', help='input dataset: mnist, cifar, svhn, fmnist')
    parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--num_class', type=int, default=10, help="class number of dataset")
    parser.add_argument('--num_channel', type=int, default=1, help="channel number of dataset: 3 or 1")
    parser.add_argument('--latent_dim', type=int, default=100, help="the dimension of latent code")
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--num_resnet', type=int, default=6, help='number of resnet blocks in generator')
    parser.add_argument('--input_size', type=int, default=28, help='input size: 32 or 28')
    parser.add_argument('--resize_scale', type=int, default=28, help='resize scale: 32 or 28')
    parser.add_argument('--crop_size', type=int, default=28, help='crop size: 32 or 28')
    parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of train epochs')
    parser.add_argument('--epoch_G', type=int, default=2, help='G number of local train epochs')
    parser.add_argument('--decay_epoch', type=int, default=20, help='start decaying learning rate after this number')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for classifier & extractor, default=0.01')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
    parser.add_argument('--lambdaA', type=float, default=1, help='lambdaA for image loss')
    parser.add_argument('--lambdaB', type=float, default=1, help='lambdaB for feature mse loss')
    parser.add_argument('--lambdaC', type=float, default=2, help='lambdaC for feature L1 loss')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--n_validation', type=int, default=1000, help="number of validation samples")

    args = parser.parse_args()
    return args
