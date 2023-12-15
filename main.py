import copy
import torch
import time
import os

from tqdm import tqdm

from PPIDSG.options import args_parser
from PPIDSG.update import LocalUpdate, test_inference
from PPIDSG.models import Generator, Discriminator, AutoEncoder_VGG, VGG16_classifier, AutoEncoder_VGG_mnist, VGG16_classifier_mnist
from PPIDSG.utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    exp_details(args)

    model_dir = args.dataset + '_model/'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        G = Generator(1, args.ngf, 1, args.num_resnet) # mnist, f-mnist:1; svhn, cifar10:3 (both input and output)
        D_B = Discriminator(1, args.ndf, 1) # mnist, f-mnist:1; svhn, cifar10:3 (only the input parameter)
        global_model = AutoEncoder_VGG_mnist().to(device)
        C = VGG16_classifier_mnist().to(device)
    else:
        G = Generator(3, args.ngf, 3, args.num_resnet)  # mnist, f-mnist:1; svhn, cifar10:3 (both input and output)
        D_B = Discriminator(3, args.ndf, 1)  # mnist, f-mnist:1; svhn, cifar10:3 (only the input parameter)
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

    # Training
    global_model_local_weights = [global_weights for i in range(args.num_users)]
    D_B_local_weights = [D_B_weights for i in range(args.num_users)]
    C_local_weights = [C_weights for i in range(args.num_users)]

    # federated learning
    for epoch in tqdm(range(args.num_epochs)):
        G_local_weights = []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        for idx in range(args.num_users):
            global_model.load_state_dict(global_model_local_weights[idx]) # local train
            D_B.load_state_dict(D_B_local_weights[idx]) # local train
            C.load_state_dict(C_local_weights[idx]) # local train

            local_model = LocalUpdate(args=args, dataset=train_dataset, G=copy.deepcopy(G),
                                      D_B=copy.deepcopy(D_B), idxs=user_groups[idx])
            w, v, u, z = local_model.update_weights(D_A_model=copy.deepcopy(global_model),
                                                    C=copy.deepcopy(C), global_round=epoch)
            G_local_weights.append(copy.deepcopy(v))
            D_B_local_weights[idx] = copy.deepcopy(u) # discriminator
            global_model_local_weights[idx] = copy.deepcopy(w) # feature extractor

        G_weights = average_weights(G_local_weights) # federated train
        G.load_state_dict(G_weights) # each client generator

        # test accuracy
        test_acc = test_inference(G, global_model, C, test_dataset)
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
