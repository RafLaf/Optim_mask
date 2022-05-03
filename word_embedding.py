from logging import raiseExceptions
import torch
from PIL import Image
import numpy as np
from args import args



file = str(args.labels)
f=open(file,"r")
f.readline()
lines=f.readlines()
result=[]
code = []
for x in lines:
    try:
        result.append(x.split('	')[1][:-1])
        code.append(x.split('	')[0])
    except:
        pass
f.close()

import pickle
open_file = open( str(args.codes), "rb")
L_index = pickle.load(open_file)
open_file.close()

labels = []
for i in range(len(L_index)):
    for x in L_index[i]:
        labels.append(result[code.index(x)])



def run_classes_sample(text_feat,n_ways, dmax,n_runs, distances=None,maxiter = 1000,label=labels):
    run_classes = torch.zeros(n_runs, n_ways)
    k=0
    itera = 0 
    while k<n_runs and itera<maxiter:
        itera+=1
        i = torch.randint(0,text_feat.shape[0],(1,))
        dist = distances[i]
        d = torch.where(dist<dmax)[1]
        if d.shape[0]>=n_ways:
            ind = torch.randperm(d.shape[0])[:n_ways]
            out = d[torch.randperm(d.shape[0])[:n_ways]]
            run_classes[k] = out
            k+=1
            hu = [label[i] for i in out]
            itera = 0
            print(hu)
    if itera!=maxiter:
        return run_classes
    else:
        raise ValueError("dmax too low")





