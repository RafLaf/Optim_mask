import argparse
import os
import random
import numpy as np


parser = argparse.ArgumentParser(description="""bonjour
""", formatter_class=argparse.RawTextHelpFormatter)

### hyperparameters

parser.add_argument("--lr", type=float, default="0.001", help="initial learning rate (negative is for Adam, e.g. -0.001)")
parser.add_argument("--loss-fn", type=str, default="ncm_loss", help="device(s) to use, for multiple GPUs try cuda:ijk, will not work with 10+ GPUs")
parser.add_argument("--eval-fn", type=str, default="ncm", help="device(s) to use, for multiple GPUs try cuda:ijk, will not work with 10+ GPUs")


### pytorch options
parser.add_argument("--device", type=str, default="cuda:0", help="device(s) to use, for multiple GPUs try cuda:ijk, will not work with 10+ GPUs")

### run options
parser.add_argument("--wd", type=float, default=0.01, help="weight decay")

parser.add_argument("--elts-class", type=str, default="/users/local/datasets/tieredimagenet/num_elements.pt", help="test features and exit")

parser.add_argument("--test-features", type=str, default='/users/local/r21lafar/features/tiered/tieredfeaturesAS2.pt11', help="test features and exit")
parser.add_argument("--semantic-features", type=str, default='/users/local/r21lafar/features/tiered/tiered_semantic_features.pt', help="load text features")
parser.add_argument("--labels", type=str, default='/users/local/datasets/labels_tiered.txt', help="load labels")
parser.add_argument("--codes", type=str, default='/users/local/datasets/Tiered_codes.txt', help="load codes")
parser.add_argument("--wandb", type=str, default='', help="Report to wandb, input is the entity name")

### few-shot parameters
parser.add_argument("--n-shots", type=int, default=5, help="how many shots per few-shot run, can be int or list of ints. In case of episodic training, use first item of list as number of shots.")
parser.add_argument("--n-runs", type=int, default=100, help="number of few-shot runs")
parser.add_argument("--n-ways", type=int, default=5, help="number of few-shot ways")
parser.add_argument("--n-queries", type=int, default=150, help="number of few-shot queries")

parser.add_argument("--ortho" ,action="store_true", help="create hard problems")
parser.add_argument("--semantic-difficulty" ,action="store_true", help="create hard problems")
parser.add_argument("--masking" ,action="store_true", help="create hard problems")

parser.add_argument("--transductive-temperature-softkmeans", type=float, default=5, help="temperature for few-shot transductive is using softkmeans")

try :
    get_ipython()
    args = parser.parse_args(args=[])
except :
    args = parser.parse_args()



if args.device[:5] == "cuda:" and len(args.device) > 5:
    args.devices = []
    for i in range(len(args.device) - 5):
        args.devices.append(int(args.device[i+5]))
    args.device = args.device[:6]
else:
    args.devices = [args.device]


