import os
import argparse
import torch
import numpy as np
import torchvision.datasets
from torch.utils.data import DataLoader
from evaluation import evaluation
import copy


from modules import resnet, transform, simsam, contrastive_loss
from utils import yaml_config_hook, save_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def inference():
    model.eval()
    feature_vector = []
    labels_vector = []

    for step, data in enumerate(data_loader):
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
            transform=transform.Transforms(size=args.img_size).test_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.img_size).test_transform
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num=10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.img_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.img_size).test_transform,
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == 'STL-10':
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            download=True,
            split='train',
            transform=transform.Transforms(size=args.img_size).test_transform
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            download=True,
            split='test',
            transform=transform.Transforms(size=args.img_size).test_transform
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num=10
    elif args.dataset == "tiny-ImageNet":
        train_dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(size=args.img_size).test_transform,
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/val',
            transform=transform.Transforms(size=args.img_size).test_transform,
        )
        class_num = 200
    elif args.dataset == 'ImageNet-10':
        train_dataset = torchvision.datasets.ImageFolder(root=args.dataset_dir + '/ImageNet10/train', transform=transform.Transforms(size=args.img_size).test_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=args.dataset_dir + '/ImageNet10/test', transform=transform.Transforms(size=args.img_size,).test_transform)
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num=10
    elif args.dataset == 'ImageNet-dogs':
        train_dataset = torchvision.datasets.ImageFolder(root=args.dataset_dir + '/ImageNetDogs/train',
                                                         transform=transform.Transforms(size=args.img_size).test_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=args.dataset_dir + '/ImageNetDogs/test',
                                                        transform=transform.Transforms(size=args.img_size).test_transform)
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 15
    else:
        raise NotImplementedError

    data_loader = DataLoader(
        dataset= test_dataset, # ImageNet is dataset
        batch_size=2000,
        shuffle=False,
        drop_last=False,
        num_workers=args.worker
    )

    res = resnet.get_resnet(args.resnet)
    model = simsam.SimSiam(res, args.feature_dim, class_num)
    model = model.to(device)
    model_fp = os.path.join(args.model_path, "CIFAR-10_checkpoint.tar")
    
    model.load_state_dict(torch.load(model_fp, map_location=device)['net'])
    model = model.to(device)
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
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))






