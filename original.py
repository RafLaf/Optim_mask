import torch
import torch.nn as nn


num_classes = 5
num_shots = 5
num_queries = 100

# loading / preprocessing the dataset
dataset = torch.load("/home/tesbed/datasets/easy/feats/resnet12/minifeatures1.pt11", map_location=torch.device("cpu"))
shape = dataset.shape

average = dataset[:64].reshape(-1, dataset.shape[-1]).mean(dim = 0)

dataset = dataset.reshape(-1, dataset.shape[-1]) - average

dataset = dataset / torch.norm(dataset, dim = 1, keepdim = True)

dataset = dataset.reshape(shape)



# identifying base classes contributions

ini_centroids = dataset[:64].mean(dim = 1)

u, _, v = torch.svd(ini_centroids)

centroids = torch.matmul(u, v.transpose(0,1))

torch.einsum("nd,nd->n", ini_centroids / torch.norm(ini_centroids,dim = 1, keepdim = True), centroids)

#print(centroids.shape)

# masking model

class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.mask = nn.Parameter(torch.zeros(64))

    def forward(self, x):
        contribs = torch.einsum("csd,bd->csb", x, centroids)
        remove_contribs = torch.clamp(self.mask.unsqueeze(0).unsqueeze(0), 0, 1) * contribs
        return x - torch.einsum("csb,bd->csd", remove_contribs, centroids)

# compute snr

def snr(x, num_classes = num_classes):
    snrs = 0.
    for i in range(num_classes):
        for j in range(i, num_classes):
            margin = torch.norm(x[i].mean(dim = 0) - x[j].mean(dim = 0), dim = 0)
            std = 0.5 * (torch.norm(x[i] - x[i].mean(dim = 0, keepdim = True), dim = 1).mean() + torch.norm(x[j] - x[j].mean(dim = 0, keepdim = True), dim = 1).sum())
            snrs += margin / std
    return snrs / (num_classes * (num_classes - 1)) * 2

def ncm_loss(run):
    centroids = run[:,:num_shots].mean(dim = 1)
    dists = torch.norm(run[:,:].unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0), dim = 3)
    sims = (-1 * dists).reshape(-1, run.shape[0])
    targets = torch.repeat_interleave(torch.arange(run.shape[0]), dists.shape[1])
    return nn.CrossEntropyLoss()(sims, targets)


# generation of runs

def generate_run(num_classes = num_classes, num_shots = num_shots, num_queries = num_queries):
    classes = torch.randperm(20)[:num_classes] + 80
    run = torch.zeros(num_classes, num_shots + num_queries, dataset.shape[-1])
    for i in range(num_classes):
        run[i] = dataset[classes[i]][torch.randperm(dataset.shape[1])[:num_shots + num_queries]]
    return run

# ncm

def ncm(run, num_shots = num_shots):
    centroids = run[:,:num_shots].mean(dim = 1)
    dists = torch.norm(run[:,num_shots:].unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0), dim = 3)
    mins = torch.min(dists, dim = 2)[1]
    return (mins == torch.arange(run.shape[0]).unsqueeze(1)).float().mean()

def test(n_tests, wd = 0):
    pre = 0.
    post = 0.
    for test in range(n_tests):
        run = generate_run()
        mask = Mask()
        optimizer = torch.optim.Adam(mask.parameters(), lr = 1e-3,  weight_decay = wd)
        pre += ncm(run)

        for i in range(1000):
            optimizer.zero_grad()
            loss = ncm_loss(mask(run[:,:num_shots]))
            loss.backward()
            optimizer.step()

        post += ncm(mask(run))
        print("\r{:3d}% {:.4f} {:.4f}".format(int(100 * (test+1) / n_tests), pre.item() / (test+1), post.item() / (test+1)), end = '')
    print("\r{:.4f} {:.4f}      ".format(pre.item() / n_tests, post.item() / n_tests))

import sys
test(int(sys.argv[1]), wd = float(sys.argv[2]))