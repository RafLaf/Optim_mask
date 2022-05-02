from sklearn import semi_supervised
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from word_embedding import *
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes = 5
num_shots = 2
num_queries =58
dim = 640
T = int(sys.argv[3])
print("nways = {}, n_shots = {}, T={}".format(num_classes,num_shots,T))


# loading / preprocessing the dataset
try:
    dataset = torch.load('/users/local/r21lafar/features/tiered/tieredfeatures2.pt11', map_location=torch.device(device))
except:
    dataset = torch.load(str(sys.argv[6]), map_location=torch.device(device))
#shape = dataset.shape
semantic_features = torch.load('/users/local/r21lafar/features/tiered/tiered_semantic_features.pt', map_location=torch.device(device))
st_novel = 351 + 97
print(dataset.shape)
semantic_features_n = semantic_features[st_novel:] 
distances_n = torch.cdist(semantic_features_n,semantic_features_n)
print(semantic_features_n.shape, distances_n.shape)

if False:
    feat_train = dataset['base']
    feat_val = dataset['val']
    feat_novel = dataset['novel']
    average = feat_train[:64].reshape(-1, feat_train.shape[-1]).mean(dim = 0)
    s = feat_novel.shape
    feat_novel = feat_novel.reshape(-1, feat_novel.shape[-1]) - average
    feat_novel = feat_novel / torch.norm(feat_novel, dim = 1, keepdim = True)
    feat_novel = feat_novel.reshape(s)
    ini_centroids = feat_train[:64].mean(dim = 1)


else:
    shape = dataset.shape
    average = dataset[:64].reshape(-1, dataset.shape[-1]).mean(dim = 0)
    dataset = dataset.reshape(-1, dataset.shape[-1]) - average
    dataset = dataset / torch.norm(dataset, dim = 1, keepdim = True)
    dataset = dataset.reshape(shape)
    ini_centroids = dataset[:64].mean(dim = 1)



# identifying base classes contributions


u, _, v = torch.svd(ini_centroids)

centroids = torch.matmul(u, v.transpose(0,1))
#centroids = torch.randn(64,640).to(device)*0.04
#torch.einsum("nd,nd->n", ini_centroids / torch.norm(ini_centroids,dim = 1, keepdim = True), centroids)


# masking model

class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.mask = nn.Parameter(torch.ones(64)*0.5)
    def forward(self, x):
        contribs = torch.einsum("csd,bd->csb", x, centroids)
        remove_contribs = torch.clamp(self.mask.unsqueeze(0).unsqueeze(0), 0, 1) * contribs
        return x - torch.einsum("csb,bd->csd", remove_contribs, centroids)


st_novel = 351 + 97 


def generate_run(num_classes = num_classes, num_shots = num_shots, num_queries = num_queries, dmax= 6.5 ,label= labels[st_novel:]):
    classes = run_classes_sample(semantic_features_n,n_ways = num_classes, dmax=dmax,n_runs = 1 , distances=distances_n,maxiter = 1000, label = label).long() + st_novel
    run = torch.zeros(num_classes, num_shots + num_queries, dataset.shape[-1]).to(device)
    for i in range(num_classes):
        run[i] = dataset[classes[0][i]][torch.randperm(dataset.shape[1])[:num_shots + num_queries]]
    return run
# ncm


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
    targets = torch.repeat_interleave(torch.arange(run.shape[0]), dists.shape[1]).to(device)
    return nn.CrossEntropyLoss()(sims, targets)


# generation of runs

def ncm(run, num_shots = num_shots):
    centroids = run[:,:num_shots].mean(dim = 1)
    dists = torch.norm(run[:,num_shots:].unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0), dim = 3)
    mins = torch.min(dists, dim = 2)[1]
    return (mins == torch.arange(run.shape[0]).unsqueeze(1).to(device)).float().mean()


def transductive_ncm_loss(run,num_shots = num_shots ):
    with torch.no_grad():
        means = torch.mean(run[:,:num_shots], dim = 1)
        for i in range(30):
            similarities = torch.norm(run[:,num_shots:].reshape( -1, 1, run.shape[-1]) - means.reshape( 1, num_classes, run.shape[-1]), dim = 2, p = 2)
            soft_allocations = F.softmax(-similarities.pow(2)*T,dim =1)
            means = torch.sum(run[:,:num_shots], dim = 1) + torch.einsum("sw,sd->wd", soft_allocations, run[:,num_shots:].reshape(-1, run.shape[2]))
            means = means/(num_shots+soft_allocations.sum(dim = 0).reshape(-1, 1))
    dists = torch.norm(run[:,:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim = 3)
    sims = (-1 * dists).reshape(-1, run.shape[0])
    targets = torch.repeat_interleave(torch.arange(run.shape[0]), dists.shape[1]).to(device)
    return  nn.CrossEntropyLoss()(sims, targets)


def transductive_loss(run,num_shots = num_shots ):
    #with torch.no_grad():
    means = torch.mean(run[:,:num_shots], dim = 1)
    for i in range(30):
        similarities = torch.norm(run[:,num_shots:].reshape( -1, 1, run.shape[-1]) - means.reshape( 1, num_classes, run.shape[-1]), dim = 2, p = 2)
        soft_allocations = F.softmax(-similarities.pow(2)*T,dim =1)
        means = torch.sum(run[:,:num_shots], dim = 1) + torch.einsum("sw,sd->wd", soft_allocations, run[:,num_shots:].reshape(-1, run.shape[2]))
        means = means/(num_shots+soft_allocations.sum(dim = 0).reshape(-1, 1))
    dists = torch.norm(run[:,:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim = 3)
    sims = (-1 * dists).reshape(-1, run.shape[0])
    targets = torch.repeat_interleave(torch.arange(run.shape[0]), dists.shape[1]).to(device)
    return  nn.CrossEntropyLoss()(sims, targets)

def kmeans(run,num_shots = num_shots ):
    with torch.no_grad():
        means = torch.mean(run[:,:num_shots], dim = 1)
        for i in range(30):
            similarities = torch.norm(run[:,num_shots:].reshape( -1, 1, run.shape[-1]) - means.reshape( 1, num_classes, run.shape[-1]), dim = 2, p = 2)
            soft_allocations = F.softmax(-similarities.pow(2)*T,dim =1)
            means = torch.sum(run[:,:num_shots], dim = 1) + torch.einsum("sw,sd->wd", soft_allocations, run[:,num_shots:].reshape(-1, run.shape[2]))
            means = means/(num_shots+soft_allocations.sum(dim = 0).reshape(-1, 1))
    dists = torch.norm(run[:,num_shots:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim = 3)
    mins = torch.min(dists, dim = 2)[1]
    return (mins == torch.arange(run.shape[0]).unsqueeze(1).to(device)).float().mean()


def kmeans_alloc(run,num_shots = num_shots ):
    with torch.no_grad():
        means = torch.mean(run[:,:num_shots], dim = 1)
        for i in range(30):
            similarities = torch.norm(run[:,num_shots:].reshape( -1, 1, run.shape[-1]) - means.reshape( 1, num_classes, run.shape[-1]), dim = 2, p = 2)
            soft_allocations = F.softmax(-similarities.pow(2)*T,dim =1)
            means = torch.sum(run[:,:num_shots], dim = 1) + torch.einsum("sw,sd->wd", soft_allocations, run[:,num_shots:].reshape(-1, run.shape[2]))
            means = means/(num_shots+soft_allocations.sum(dim = 0).reshape(-1, 1))
    support = run[:,:num_shots]
    queries = run[:,num_shots:].reshape(-1,run.shape[-1])
    return support, queries , soft_allocations



def sil_score_coorected(support,queries, soft_allocation):
    s = support.shape
    support_reshaped  =support.reshape( -1 , s[-1])
    samples = torch.cat((support_reshaped, queries), dim =0 )
    target = torch.zeros(support_reshaped.shape[0], num_classes).to(device)
    for i in range(num_classes):
        for j in range(num_shots):
            target[i*num_shots + j,i]=1
    soft_alloc = torch.cat((target, soft_allocation), dim = 0)
    soft_allocations_normalized = soft_alloc/ soft_alloc.sum(0, keepdim= True)
    coef = torch.einsum('jl, ik -> jilk', soft_allocations_normalized,soft_allocations_normalized)
    norm = torch.sum(coef, dim = (0,1) ) 
    coef = coef/norm
    coef_a = torch.eye(coef.shape[-1]).to(device)
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
    support, queries , soft_allocations = kmeans_alloc(run)
    s = sil_score_coorected(support,queries, soft_allocations)
    return -s


L_inductive = [snr, ncm_loss]

def test(n_tests, wd = 0, loss_fn =ncm_loss, eval_fn = ncm):
    print(loss_fn, eval_fn)
    print('wd = {}'.format(wd))
    pre = 0.
    post = 0.
    if loss_fn in L_inductive:
        print('no cheat inductive')
    print('')
    for test in range(n_tests):
        run = generate_run()
        mask = Mask().to(device)
        optimizer = torch.optim.Adam(mask.parameters(), lr = 1e-3,  weight_decay = wd)
        #pre += ncm(run)
        pre += eval_fn(run)
        for i in range(1000):
            optimizer.zero_grad()
            if loss_fn in L_inductive:
                loss = loss_fn(mask(run[:,:num_shots]))
            else:
                loss = loss_fn(mask(run))
            loss.backward()
            optimizer.step()

            

        post += eval_fn(mask(run))
        print("\r{:3d}% {:.4f} {:.4f} {:.4f}".format(int(100 * (test+1) / n_tests), pre.item() / (test+1), post.item() / (test+1),(post.item()-pre.item()) / (test+1)), end = '')
    print("\r{:.4f} {:.4f}   {:.4f}   ".format(pre.item() / n_tests, post.item() / n_tests,(post.item()-pre.item()) / n_tests) )

#alloc = transductive(run)
test(int(sys.argv[1]), wd = float(sys.argv[2]), loss_fn = eval(sys.argv[4]), eval_fn = eval(sys.argv[5]))
