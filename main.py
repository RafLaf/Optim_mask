from sklearn import semi_supervised
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from args import args
from loss_eval import *
import numpy as np
if args.wandb!='':
    import wandb
    tag =[]
    if args.masking:
        tag.append ('masking')
    else:
        tag.append('iterate')
    if args.ortho:
        tag.append('ortho')
    wandb.init(project="optim", 
            entity=args.wandb, 
            tags=tag, 
            group = 'std',
            notes=str(vars(args))
            )


print(args)

if args.semantic_difficulty:
    from word_embedding import *
    print('semantic difficulty ACTIVATED')

torch.manual_seed(0)
device = args.device

num_classes = args.n_ways
num_shots = args.n_shots
num_queries = args.n_queries
T = args.transductive_temperature_softkmeans
print("nways = {}, n_shots = {}, T={}".format(num_classes,num_shots,T))

# loading / preprocessing the dataset

dataset = torch.load(str(args.test_features), map_location=torch.device(device))
if torch.is_tensor(dataset):
    print(f'{dataset.shape=}')

    if dataset.shape[0]==100:
        elements_per_class = [600]*100
        nb_base = 64
        nb_val = 16
        nb_novel = 20
    elif dataset.shape[0]==608:
        elements_per_class = torch.load(args.elts_class)
        elements_per_class = elements_per_class['train']+elements_per_class['val']+elements_per_class['test']
        nb_base = 351
        nb_val = 97
        nb_novel = 160
    else:
        raise ValueError('features not accepted (only mini and tiered)')
    
    shape = dataset.shape
    average = torch.stack([dataset[i, :int(elements_per_class[i])].mean(dim = 0) for i in range(nb_base)]).mean(dim = 0)
    dataset = dataset.reshape(-1, dataset.shape[-1]) - average
    normE = torch.norm(dataset, dim = 1, keepdim = True)
    dataset = dataset / normE
    dataset = dataset.reshape(shape)
    centroids =  torch.stack([dataset[i,:elements_per_class[i]].mean(dim=0) for i in range(dataset.shape[0])])
    if args.use_classifier:
        model = torch.load(args.test_model, map_location =args.device)
        classifier = model['linear.weight']
        if args.MEclassifier:
            norm = torch.norm(classifier, dim= 1)
            print(f'{classifier.shape=} {average.shape=} {norm.shape=}')
            centroids['train'] = (classifier - average)/norm.unsqueeze(1).repeat(1,shape[-1])
        else:
            centroids['train'] = classifier


    elif args.random:
        centroids['train'] = torch.randn(centroids[:nb_base].shape)

    novel_features = dataset[-nb_novel:]
    dim = dataset.shape[-1]
    assert(num_shots+num_queries<novel_features.shape[-1])

else:
    elements_per_class = torch.load(args.elts_class)
    base_features = dataset['base']
    val_features = dataset['val']
    novel_features = dataset['novel']
    nb_base = base_features.shape[0]
    nb_val = val_features.shape[0]
    nb_novel = novel_features.shape[0]
    average = base_features[:nb_base].reshape(-1, base_features.shape[-1]).mean(dim = 0)
    s = novel_features.shape
    novel_features = novel_features.reshape(-1, novel_features.shape[-1]) - average
    novel_features = novel_features / torch.norm(novel_features, dim = 1, keepdim = True)
    novel_features = novel_features.reshape(s)
    centroids ={}
    centroids['train'] = torch.stack([base_features[i,:elements_per_class['train'][i]].mean(dim=0) for i in range(base_features.shape[0])])
    centroids['val'] = torch.stack([val_features[i,:elements_per_class['val'][i]].mean(dim=0) for i in range(val_features.shape[0])])
    centroids['test'] = torch.stack([novel_features[i,:elements_per_class['test'][i]].mean(dim=0) for i in range(novel_features.shape[0])])

    if args.use_classifier:
        model = torch.load(args.test_model, map_location =args.device)
        centroids[:nb_base] = model['linear.weight']
    dataset_n = novel_features
    assert (not args.semantic_difficulty)
    dim = base_features.shape[-1]
    assert(num_shots+num_queries<novel_features.shape[-1])


if args.ortho:
    u, _, v = torch.svd(centroids['train'])
    centroids['train'] = torch.matmul(u, v.transpose(0,1))  #orthogonalization


if args.semantic_difficulty:
    semantic_features = torch.load(args.semantic_features, map_location=torch.device(device))
    semantic_features_n = semantic_features[nb_base+nb_val:] 
    if semantic_features_n.shape[0] != nb_novel:
        print(f'{semantic_features_n.shape =} {dataset.shape=}')
        raise ValueError('PLEASE MAKE SURE semantic features correspond to visual features (NOT the same size here)')
    distances_n = torch.cdist(semantic_features_n,semantic_features_n)


def project(run, i):
    ncentroid = centroids['train'][i]
    ncentroid = ncentroid / torch.norm(ncentroid, dim = 0)
    run = run - torch.einsum("csd,d->cs", run, ncentroid).unsqueeze(2) * ncentroid.unsqueeze(0).unsqueeze(0)
    return run / torch.norm(run, dim = 2, keepdim = True)





# masking model

class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.mask = nn.Parameter(torch.ones(nb_base)*0.1)
    def forward(self, x):
        contribs = torch.einsum("csd,bd->csb", x, centroids[:nb_base])
        remove_contribs = torch.clamp(self.mask.unsqueeze(0).unsqueeze(0), 0, 1) * contribs
        return x - torch.einsum("csb,bd->csd", remove_contribs, centroids[:nb_base])



def generate_run(num_classes = num_classes, num_shots = num_shots, num_queries = num_queries, dmax= 6.5 ,label=None ):
    if args.semantic_difficulty:
        classes = run_classes_sample(semantic_features_n,n_ways = num_classes, dmax=dmax,n_runs = 1 , distances=distances_n,maxiter = 1000, label = labels[nb_base+nb_val:] ).long()
    else:
        classes = torch.randperm(nb_novel)[:num_classes]
    samples = []
    for i in range(num_classes):
        samples.append(novel_features[classes[i], torch.randperm(elements_per_class['test'][classes[i]])[:num_shots+num_queries]])
    return torch.max(torch.norm(centroids['test'] [classes].unsqueeze(0) - centroids['test'] [classes].unsqueeze(1))), torch.stack(samples)
# ncm


# compute snr

L_inductive = [snr, ncm_loss]

def test(n_tests,wd = 0, loss_fn =ncm_loss, eval_fn = ncm, masking =args.masking, T = args.transductive_temperature_softkmeans, lr=args.lr, num_queries = args.n_queries):
    print(loss_fn, eval_fn)
    if args.masking:
        print('wd = {}'.format(wd))
    pre = []
    post = []
    if loss_fn in L_inductive:
        print('no cheat inductive')
    print('')
    selectivities =[]
    results = {'n_queries': num_queries}
    for test in range(n_tests):
        selectivity,run = generate_run(num_queries = num_queries)
        selectivities.append(selectivity)
        pre.append(eval_fn(run).item())
        if masking:
            mask = Mask().to(device)
            optimizer = torch.optim.Adam(mask.parameters(), lr = lr,  weight_decay = wd)
            for i in range(1000):
                optimizer.zero_grad()
                if loss_fn in L_inductive:
                    loss = loss_fn(mask(run[:,:num_shots]))
                else:
                    loss = loss_fn(mask(run))
                #print(loss.item())
                loss.backward()
                optimizer.step()
            post.append( eval_fn(mask(run)).item())
            #print(mask.mask.sort())
        elif args.greedy:
            current_confidence = loss_fn(run)
            for i in range(nb_base):
                new_run = project(run, i)
                new_confidence = loss_fn(new_run)
                if new_confidence > current_confidence:
                    current_confidence = new_confidence
                    run = new_run
            post.append(eval_fn(run).item())
        else:
            current_confidence = -loss_fn(run)
            L_good=[]
            for i in range(nb_base):
                new_run = project(run, i)
                new_confidence = -loss_fn(new_run)
                if new_confidence > current_confidence:
                    L_good.append(i)
            for j in L_good:
                run = project(run, j)
            post.append(eval_fn(run).item())
        
        print("\r", end='')
        for name,indexes in [("all", np.arange(test + 1)), ("hard",np.where(selectivities < mean - std)[0]), ("easy",np.where(selectivities > mean + std)[0])]:      

            if len(indexes) > 0:
                results[name+'_len']= len(indexes)
                results[name+'_post']=np.mean(np.array(post)[indexes])
                results[name+'_boost']=( np.mean(np.array(post)[indexes]- np.mean(np.array(pre)[indexes])))
                results[name+'_boostSTD']=( np.std(np.array(post)[indexes]- np.mean(np.array(pre)[indexes])))
                print("{:s} ({:4d}) {:.2f}% (boost: {:.2f}%) ".format(name, len(indexes), 100 * np.mean(np.array(post)[indexes]), 100 * (np.mean(np.array(post)[indexes]) - np.mean(np.array(pre)[indexes]))), end='')
        print("    ", end ='')
    print()
    return results

selectivities = []
for _ in range(1000):
    selectivity, run = generate_run()
    selectivities.append(selectivity.item())
mean, std = np.mean(selectivities), np.std(selectivities)


list_wd = np.logspace(-4,-1,3)
list_lr = np.logspace(-2,-3,3)
list_queries =np.linspace(5,100,4).astype(int)
if args.parameter_scan:

    if args.masking:
        for wd in list_wd:
            for lr in list_lr:
                results = test(int(args.n_runs), wd = wd, loss_fn = eval(args.loss_fn), eval_fn = eval(args.eval_fn), lr = lr)
                
                results['wd'] = wd
                results['lr'] = lr

                wandb.log(results)
    else:
        for n_queries in list_queries:
            selectivities = []
            for _ in range(1000):
                selectivity, run = generate_run(num_queries=n_queries)
                selectivities.append(selectivity.item())
            mean, std = np.mean(selectivities), np.std(selectivities)
            results = test(int(args.n_runs),  loss_fn = eval(args.loss_fn), eval_fn = eval(args.eval_fn), num_queries = n_queries)
            wandb.log(results)
else:
    results = test(n_tests = int(args.n_runs), loss_fn = eval(args.loss_fn), eval_fn = eval(args.eval_fn), lr = args.lr, wd = args.wd)