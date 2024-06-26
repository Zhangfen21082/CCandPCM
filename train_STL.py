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

import copy


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
            # c = model.module.forward_cluster(x)
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(data_loader_test)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def train():
    model.train()
    loss_epoch = 0
    
    
    # unlabeled data
    for step, ((x_i, x_j, x_s), _) in enumerate(instance_data_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        x_s = x_s.to(device)

        p_i, p_j, p_s, p_instance_i, p_instance_j, p_instance_s, c_i, c_j, c_s = model(x_i, x_j, x_s)
        loss_instance0 = contrastive_loss.loss_fn(p_i, p_instance_j.detach()) * 0.5 + contrastive_loss.loss_fn(p_j, p_instance_i.detach()) * 0.5
        loss_instance1 = contrastive_loss.loss_fn(p_s, p_instance_j.detach()) * 0.5 + contrastive_loss.loss_fn(p_j, p_instance_s.detach()) * 0.5
        loss_instance = loss_instance0 + loss_instance1
        
        loss = loss_instance
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(instance_data_loader)}]\t loss_instance: {loss_instance.item()}")

            
        loss_epoch += loss.item()

    for step, ((x_i, x_j, x_s), _) in enumerate(cluster_data_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        x_s = x_s.to(device)

        p_i, p_j, p_s, p_instance_i, p_instance_j, p_instance_s, c_i, c_j, c_s = model(x_i, x_j, x_s)
        loss_instance0 = contrastive_loss.loss_fn(p_i, p_instance_j.detach()) * 0.5 + contrastive_loss.loss_fn(p_j, p_instance_i.detach()) * 0.5
        loss_instance1 = contrastive_loss.loss_fn(p_s, p_instance_j.detach()) * 0.5 + contrastive_loss.loss_fn(p_j, p_instance_s.detach()) * 0.5


        loss_cluster0 = criterion_cluster(c_i, c_j.detach()) * 0.5 + criterion_cluster(c_j, c_i.detach()) * 0.5
        loss_cluster1 = criterion_cluster(c_s, c_j.detach()) * 0.5 + criterion_cluster(c_j, c_s.detach()) * 0.5
        



        loss_instance = loss_instance0 + loss_instance1
        loss_cluster = loss_cluster0 + loss_cluster1
        loss = loss_instance + loss_cluster * args.sigma
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(cluster_data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    
    for step, ((x_i, x_j, x_s), _) in enumerate(cluster_data_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        x_s = x_s.to(device)
        c_i, c_j, c_s = model.forward_pui(x_i, x_j, x_s)
        loss_pui = pui(c_j, c_i) + pui(c_j, c_s)
        loss_pui.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(cluster_data_loader)}]\t loss_pui: {loss_pui.item()}")
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

   
    if args.dataset == 'STL-10':
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.img_size),
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.img_size),
        )
        unlabeled_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=transform.Transforms(size=args.img_size),
        )
        test_dataset_test = torchvision.datasets.STL10(
            root=args.dataset_dir,
            download=True,
            split='test',
            transform=transform.Transforms(size=args.img_size).test_transform
        )
        cluster_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        instance_dataset = unlabeled_dataset
        class_num = 10
    else:
        raise NotImplementedError

    cluster_data_loader = torch.utils.data.DataLoader(
        cluster_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.worker,
    )
    instance_data_loader = torch.utils.data.DataLoader(
        instance_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.worker,
    )

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

    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(loss_device)
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
            nmi, ari, f, acc = evaluation.evaluate(y, X)
            if acc > max_acc:
                max_acc = acc
                max_epoch = epoch
            print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f} MAX_ACC = {:.4f} MAX_EPOCH = {:.4f}'.format(nmi, ari, f,
                                                                                                           acc, max_acc,
                                                                                                           max_epoch))
            save_model(args, model, optimizer, epoch)

        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(instance_data_loader)}\t Time:{(train_epoch_end - train_epoch_start) / 60.0}\t Total_Time:{total_time}")


