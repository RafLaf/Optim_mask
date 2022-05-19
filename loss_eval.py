import torch
from args import args
import torch.nn.functional as F
T = args.transductive_temperature_softkmeans


def snr(x, num_classes = args.n_ways):
    snrs = 0.
    for i in range(num_classes):
        for j in range(i, num_classes):
            margin = torch.norm(x[i].mean(dim = 0) - x[j].mean(dim = 0), dim = 0)
            std = 0.5 * (torch.norm(x[i] - x[i].mean(dim = 0, keepdim = True), dim = 1).mean() + torch.norm(x[j] - x[j].mean(dim = 0, keepdim = True), dim = 1).sum())
            snrs += margin / std
    return snrs / (num_classes * (num_classes - 1)) * 2




def ncm_loss(run):
    centroids = run[:,:args.n_shots].mean(dim = 1)
    dists = torch.norm(run[:,:].unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0), dim = 3)
    sims = (-1 * dists).reshape(-1, run.shape[0])
    targets = torch.repeat_interleave(torch.arange(run.shape[0]), dists.shape[1]).to(args.device)
    return torch.nn.CrossEntropyLoss()(sims, targets)


# generation of runs

def ncm(run, num_shots = args.n_shots, confidence = False):
    centroids = run[:,:num_shots].mean(dim = 1)
    dists = torch.norm(run[:,num_shots:].unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0), dim = 3)
    if confidence:
        sims = torch.softmax((-5 * dists).reshape(-1, run.shape[0]), dim = 1)
        return torch.max(sims, dim = 1)[0].mean()
    else:
        mins = torch.min(dists, dim = 2)[1]
        return (mins == torch.arange(run.shape[0]).unsqueeze(1).to(args.device)).float().mean()

def ncm_confidence(run):
    return -ncm(run, confidence = True)


def transductive_ncm_loss(run,num_shots = args.n_shots ):
    with torch.no_grad():
        means = torch.mean(run[:,:num_shots], dim = 1)
        for i in range(30):
            similarities = torch.norm(run[:,num_shots:].reshape( -1, 1, run.shape[-1]) - means.reshape( 1, args.n_ways, run.shape[-1]), dim = 2, p = 2)
            soft_allocations = F.softmax(-similarities.pow(2)*T,dim =1)
            means = torch.sum(run[:,:num_shots], dim = 1) + torch.einsum("sw,sd->wd", soft_allocations, run[:,num_shots:].reshape(-1, run.shape[2]))
            means = means/(num_shots+soft_allocations.sum(dim = 0).reshape(-1, 1))
    dists = torch.norm(run[:,:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim = 3)
    sims = (-1 * dists).reshape(-1, run.shape[0])
    targets = torch.repeat_interleave(torch.arange(run.shape[0]), dists.shape[1]).to(args.device)
    return  nn.CrossEntropyLoss()(sims, targets)


def transductive_loss(run,num_shots = args.n_shots, num_classes = args.n_ways ):
    #with torch.no_grad():
    means = torch.mean(run[:,:num_shots], dim = 1)
    for i in range(30):
        similarities = torch.norm(run[:,num_shots:].reshape( -1, 1, run.shape[-1]) - means.reshape( 1, num_classes, run.shape[-1]), dim = 2, p = 2)
        soft_allocations = F.softmax(-similarities.pow(2)*T,dim =1)
        means = torch.sum(run[:,:num_shots], dim = 1) + torch.einsum("sw,sd->wd", soft_allocations, run[:,num_shots:].reshape(-1, run.shape[2]))
        means = means/(num_shots+soft_allocations.sum(dim = 0).reshape(-1, 1))
    dists = torch.norm(run[:,:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim = 3)
    sims = (-1 * dists).reshape(-1, run.shape[0])
    targets = torch.repeat_interleave(torch.arange(run.shape[0]), dists.shape[1]).to(args.device)
    return  nn.CrossEntropyLoss()(sims, targets)

def soft_k_means(run,num_shots = args.n_shots,num_classes =args.n_ways ,alloc =False,confidence=False ):
    with torch.no_grad():
        means = torch.mean(run[:,:num_shots], dim = 1)
        for i in range(30):
            similarities = torch.norm(run[:,num_shots:].reshape( -1, 1, run.shape[-1]) - means.reshape( 1, num_classes, run.shape[-1]), dim = 2, p = 2)
            soft_allocations = F.softmax(-similarities.pow(2)*T,dim =1)
            means = torch.sum(run[:,:num_shots], dim = 1) + torch.einsum("sw,sd->wd", soft_allocations, run[:,num_shots:].reshape(-1, run.shape[2]))
            means = means/(num_shots+soft_allocations.sum(dim = 0).reshape(-1, 1))
    support = run[:,:num_shots]
    queries = run[:,num_shots:].reshape(-1,run.shape[-1])
    if alloc:
        return support, queries , soft_allocations
    dists = torch.norm(run[:,num_shots:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim = 3)
    mins = torch.min(dists, dim = 2)[1]
    if confidence:
        return (torch.max(torch.softmax(-1 * torch.pow(dists,0.5), dim = 2), dim = 2)[0]).mean()
    else:
        return (torch.argmin(dists, dim = 2) - torch.arange(5).unsqueeze(1).to(args.device) == 0).float().mean()






def sil_score_corrected(support,queries, soft_allocation, num_classes = args.n_ways, num_shots = args.n_shots ):
    s = support.shape
    support_reshaped  =support.reshape( -1 , s[-1])
    samples = torch.cat((support_reshaped, queries), dim =0 )
    target = torch.zeros(support_reshaped.shape[0], num_classes).to(args.device)
    for i in range(num_classes):
        for j in range(num_shots):
            target[i*num_shots + j,i]=1
    soft_alloc = torch.cat((target, soft_allocation), dim = 0)
    soft_allocations_normalized = soft_alloc/ soft_alloc.sum(0, keepdim= True)
    coef = torch.einsum('jl, ik -> jilk', soft_allocations_normalized,soft_allocations_normalized)
    norm = torch.sum(coef, dim = (0,1) ) 
    coef = coef/norm
    coef_a = torch.eye(coef.shape[-1]).to(args.device)
    coef_b = (coef_a-1)*(-1)
    d = torch.cdist(samples,samples)
    a_d = torch.einsum('jilk, lk -> ji', coef, coef_a )
    b_k_d = torch.einsum('jilk, lk -> jik', coef, coef_b )
    a = torch.einsum('ji, ji -> i', a_d, d)
    b_k = torch.einsum('jik, ji -> ik', b_k_d,d )
    b = torch.min(b_k , dim = -1)[0]
    s = (b-a)/torch.maximum(a,b)
    return s.mean()



def sil_loss(run):
    support, queries , soft_allocations = soft_k_means(run, alloc=True)
    s = sil_score_corrected(support,queries, soft_allocations)
    return -s


