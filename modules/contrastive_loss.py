import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PUILoss(nn.Module):

    def __init__(self, device, lamda=2.0):
        super(PUILoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.lamda = lamda
        self.device = device

    def forward(self, x, y):
        """Partition Uncertainty Index

        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """
        assert x.shape == y.shape, ('Inputs are required to have same shape')

        # partition uncertainty index
        pui = torch.mm(F.normalize(x.t(), p=2, dim=1), F.normalize(y, p=2, dim=0))
        loss_ce = self.xentropy(pui, torch.arange(pui.size(0)).to(self.device))

        # balance regularisation
        p = x.sum(0).view(-1)
        p /= p.sum()
        loss_ne = math.log(p.size(0)) + (p * p.log()).sum()

        return loss_ce + self.lamda * loss_ne

def loss_fn(p, z):
    # z = z.detach()

    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)

    return -(p*z).sum(dim=1).mean()

# def loss_fn(x, y):
#     x = F.normalize(x, dim=-1, p=2)
#     y = F.normalize(y, dim=-1, p=2)
#     return (2 - 2 * (x * y).sum(dim=-1)).mean()

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
