import torch
from args import args
import torch.nn.functional as F

dataset = torch.load(args.test_features, map_location=torch.device("cpu"))

base = 351

val = base + 97

novel = dataset.shape[0] - val

elements_per_class = torch.load(args.elts_class)
elements_per_class = elements_per_class['train']+elements_per_class['val']+elements_per_class['test']

shape = dataset.shape

mean = torch.stack([dataset[i, :elements_per_class[i]].mean(dim = 0) for i in range(base)]).mean(dim = 0)

dataset = dataset - mean.unsqueeze(0).unsqueeze(0)
dataset = dataset / torch.norm(dataset, dim = 2, keepdim = True)

n_ways = 5
n_shots = 5
n_queries = 150

centroids = torch.stack([dataset[i,:elements_per_class[i]].mean(dim=0) for i in range(dataset.shape[0])])

print(centroids[:2,:2], dataset[:2,:2,:2])
# u, _, v = torch.svd(centroids[:base])

# centroids[:base] = torch.matmul(u, v.transpose(0,1))

def generate_run(n_ways=n_ways, n_shots=n_shots, n_queries=n_queries):
    samples = []
    classes = torch.randperm(novel) + val
    for i in range(n_ways):
        samples.append(dataset[classes[i], torch.randperm(elements_per_class[classes[i]])[:n_shots+n_queries]])
    return torch.max(torch.norm(centroids[classes[:n_ways]].unsqueeze(0) - centroids[classes[:n_ways]].unsqueeze(1))), torch.stack(samples)

classes, run = generate_run()

def ncm(run, num_shots = n_shots, confidence = False):
    centroids = run[:,:num_shots].mean(dim = 1)
    dists = torch.norm(run[:,num_shots:].unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0), dim = 3)
    if confidence:
        sims = torch.softmax((-5 * dists).reshape(-1, run.shape[0]), dim = 1)
        return torch.max(sims, dim = 1)[0].mean()
    else:
        mins = torch.min(dists, dim = 2)[1]
        return (mins == torch.arange(run.shape[0]).unsqueeze(1)).float().mean()

ncm(run)

def project(run, i):
    ncentroid = centroids[i]
    ncentroid = ncentroid / torch.norm(ncentroid, dim = 0)
    run = run - torch.einsum("csd,d->cs", run, ncentroid).unsqueeze(2) * ncentroid.unsqueeze(0).unsqueeze(0)
    return run / torch.norm(run, dim = 2, keepdim = True)

import numpy as np

def soft_k_means(run, t_soft_k_means = 5, n_iter = 50, confidence = False):
    centroids = run[:,:n_shots].mean(dim = 1)
    for t in range(n_iter):
        dists = torch.pow(torch.norm(run[:,n_shots:].unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0), dim = 3), 2)
        w = torch.exp(-1 * t_soft_k_means * dists)
        w = w / torch.sum(w, dim = 2, keepdim = True)
        centroids = (run[:,:n_shots].sum(dim = 1) + torch.einsum("cqm,cqd->md", w, run[:,n_shots:])) / (n_shots + w.sum(dim = 0).sum(dim = 0).unsqueeze(1))
    if confidence:
        return (torch.max(torch.softmax(-1 * torch.pow(dists,0.5), dim = 2), dim = 2)[0]).mean()
    else:
        return (torch.argmin(dists, dim = 2) - torch.arange(5).unsqueeze(1) == 0).float().mean()

def kmeans(run,num_shots = args.n_shots,num_classes =args.n_ways ,alloc =False,confidence=False ):
    with torch.no_grad():
        means = torch.mean(run[:,:num_shots], dim = 1)
        for i in range(30):
            similarities = torch.norm(run[:,num_shots:].reshape( -1, 1, run.shape[-1]) - means.reshape( 1, num_classes, run.shape[-1]), dim = 2, p = 2)
            hard_alloc = F.one_hot(torch.argmin(similarities, dim=1), num_classes = similarities.shape[1]).float()
            means = torch.sum(run[:,:num_shots], dim = 1) + torch.einsum("sd,sw->wd",run[:,num_shots:].reshape(-1, run.shape[2]), hard_alloc)
            means = means/(num_shots+hard_alloc.sum(dim = 0).reshape(-1, 1))
    support = run[:,:num_shots]
    queries = run[:,num_shots:].reshape(-1,run.shape[-1])
    if alloc:
        return support, queries , hard_alloc
    dists = torch.norm(run[:,num_shots:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim = 3)
    mins = torch.min(dists, dim = 2)[1]
    if confidence:
        return (torch.max(torch.softmax(-1 * torch.pow(dists,0.5), dim = 2), dim = 2)[0]).mean()
    else:
        return (torch.argmin(dists, dim = 2) - torch.arange(5).unsqueeze(1).to(args.device) == 0).float().mean()


def test(confidence):
    score_before = []
    score_after = []
    selectivities = []
    for it in range(10000):
        selectivity, run = generate_run()
        # while selectivity >= mean - std and selectivity <= mean + std:
        #     selectivity, run = generate_run()
        selectivities.append(selectivity)
        score_before.append(soft_k_means(run))
        current_confidence = confidence(run)
        for i in range(base):
            new_run = project(run, i)
            new_confidence = confidence(new_run)
            if new_confidence > current_confidence:
                current_confidence = new_confidence
                run = new_run
        score_after.append(soft_k_means(run))
        print("\r", end='')
        for name,indexes in [("all", np.arange(it + 1)), ("hard",np.where(selectivities < mean - std)[0]), ("easy",np.where(selectivities > mean + std)[0])]:            
            if len(indexes) > 0:
                print("{:s} ({:4d}) {:.2f}% (boost: {:.2f}%) ".format(name, len(indexes), 100 * np.mean(np.array(score_after)[indexes]), 100 * (np.mean(np.array(score_after)[indexes]) - np.mean(np.array(score_before)[indexes]))), end='')
        print("    ", end ='')
    print()

# estimate the mean and std of selectivity of generated probs:
selectivities = []
for _ in range(1000):
    selectivity, run = generate_run()
    selectivities.append(selectivity)
mean, std = np.mean(selectivities), np.std(selectivities)
    
#cnd is the selectivity of the problem: small means very similar novel classes
#test(lambda x: ncm(x, confidence = True))
test(lambda x: soft_k_means(x, confidence = True))
#test(lambda x: kmeans(x, confidence = True))
