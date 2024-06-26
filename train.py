import os
import argparse
import torch
import numpy as np
import torchvision.datasets
import time
from torch.utils.data import DataLoader
from evaluation import evaluation
from tensorboardX import SummaryWriter

from modules import resnet, transform, contrastive_loss, simsam
from utils import yaml_config_hook, save_model, adjust_learning_rate


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference():
    model.eval()
    feature_vector = []
    labels_vector = []

    for step, data in enumerate(data_loader_test):
        x, y = data
        x = x.to(device)
        print(x.shape)
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(data_loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def train():
    model.train()
    loss_epoch = 0

    for step, ((x_i, x_j, x_s), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        x_s = x_s.to(device)

        p_i, p_j, p_s, p_instance_i, p_instance_j, p_instance_s, c_i, c_j, c_s = model(x_i, x_j, x_s)
        # instance-level
        loss_instance0 = contrastive_loss.loss_fn(p_i, p_instance_j.detach()) * 0.5 + contrastive_loss.loss_fn(p_j, p_instance_i.detach()) * 0.5
        loss_instance1 = contrastive_loss.loss_fn(p_s, p_instance_j.detach()) * 0.5 + contrastive_loss.loss_fn(p_j, p_instance_s.detach()) * 0.5

        # cluster-level
        loss_cluster0 = criterion_cluster(c_i, c_j.detach()) * 0.5 + criterion_cluster(c_j, c_i.detach()) * 0.5
        loss_cluster1 = criterion_cluster(c_s, c_j.detach()) * 0.5 + criterion_cluster(c_j, c_s.detach()) * 0.5
        



        loss_instance = loss_instance0 + loss_instance1
        loss_cluster = loss_cluster0 + loss_cluster1
        loss = loss_instance + loss_cluster * args.sigma
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    
    for step, ((x_i, x_j, x_s), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        x_s = x_s.to(device)
        c_i, c_j, c_s = model.forward_pui(x_i, x_j, x_s)
        # maximize pcl
        loss_pui = pui(c_j, c_i) + pui(c_j, c_s)
        loss_pui.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_pui: {loss_pui.item()}")
        loss_epoch += loss_pui.item()

        
            




    return loss_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook('config/config.yaml')
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'CIFAR-10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.img_size, s=0.5)
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.img_size, s=0.5)
        )
        test_dataset_test = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.img_size).test_transform
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.img_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.img_size, s=0.5),
        )
        
        test_dataset_test = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.img_size).test_transform
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == 'ImageNet-10':
        train_dataset = torchvision.datasets.ImageFolder(root=args.dataset_dir + '/ImageNet10/train',
                                                         transform=transform.Transforms(size=args.img_size, blur=True))
        test_dataset = torchvision.datasets.ImageFolder(root=args.dataset_dir + '/ImageNet10/test',
                                                        transform=transform.Transforms(size=args.img_size, blur=True))
        test_dataset_test = torchvision.datasets.ImageFolder(root=args.dataset_dir + '/ImageNet10/test', transform=transform.Transforms(size=args.img_size,).test_transform)
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == 'ImageNet-dogs':
        train_dataset = torchvision.datasets.ImageFolder(root=args.dataset_dir + '/ImageNetDogs/train',
                                                         transform=transform.Transforms(size=args.img_size, blur=True))
        test_dataset = torchvision.datasets.ImageFolder(root=args.dataset_dir + '/ImageNetDogs/test',
                                                        transform=transform.Transforms(size=args.img_size, blur=True))
        test_dataset_test = torchvision.datasets.ImageFolder(root=args.dataset_dir + '/ImageNetDogs/test',
                                                        transform=transform.Transforms(size=args.img_size).test_transform)
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        train_dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_dir + '/tiny-imagenet-200/train',
            transform=transform.Transforms(s=0.5, size=args.img_size),
        )
        test_dataset_test = torchvision.datasets.ImageFolder(
            root=args.dataset_dir + '/tiny-imagenet-200/val',
            transform=transform.Transforms(size=args.img_size).test_transform,
        )
        test_dataset_test = torchvision.datasets.ImageFolder(
            root=args.dataset_dir + '/tiny-imagenet-200/val',
            transform=transform.Transforms(size=args.img_size).test_transform,
        )
        class_num = 200
    else:
        raise NotImplementedError
    
    # for train
    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.worker,
        pin_memory=True
    )
    
    # for inference
    data_loader_test = DataLoader(
        dataset= test_dataset_test,
        batch_size=2000,
        shuffle=False,
        drop_last=False,
        num_workers=args.worker
    )

    res = resnet.get_resnet(args.resnet)
    model = simsam.SimSiam(res, args.feature_dim, class_num)
    model = model.to(device)
    init_lr = args.learning_rate * args.batch_size / 256
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device(device)

    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    pui = contrastive_loss.PUILoss(loss_device)

    total_time = 0
    print(args.start_epoch)
    
    max_acc = 0
    max_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        train_epoch_start = time.time()
        loss_epoch = train()
        train_epoch_end = time.time()
        total_time += (train_epoch_end - train_epoch_start) / 60.0
        if epoch % 10 == 0:
            print("The model is loaded and starts clustering")
            X, y = inference()
            if args.dataset == "CIFAR-100":  # super-class
                super_label = [
                    [72, 4, 95, 30, 55],
                    [73, 32, 67, 91, 1],
                    [92, 70, 82, 54, 62],
                    [16, 61, 9, 10, 28],
                    [51, 0, 53, 57, 83],
                    [40, 39, 22, 87, 86],
                    [20, 25, 94, 84, 5],
                    [14, 24, 6, 7, 18],
                    [43, 97, 42, 3, 88],
                    [37, 17, 76, 12, 68],
                    [49, 33, 71, 23, 60],
                    [15, 21, 19, 31, 38],
                    [75, 63, 66, 64, 34],
                    [77, 26, 45, 99, 79],
                    [11, 2, 35, 46, 98],
                    [29, 93, 27, 78, 44],
                    [65, 50, 74, 36, 80],
                    [56, 52, 47, 59, 96],
                    [8, 58, 90, 13, 48],
                    [81, 69, 41, 89, 85],
                ]
                y_copy = copy.copy(y)
                for i in range(20):
                    for j in super_label[i]:
                        y[y_copy == j] = i
            nmi, ari, f, acc = evaluation.evaluate(y, X)
            if acc > max_acc:
                max_acc = acc
                max_epoch = epoch
            print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f} MAX_ACC = {:.4f} MAX_EPOCH = {:.4f}'.format(nmi, ari, f, acc, max_acc, max_epoch))
            save_model(args, model, optimizer, epoch)
        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}\t Time:{(train_epoch_end - train_epoch_start) / 60.0}\t Total_Time:{total_time}")
    
        


