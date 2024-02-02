#!/usr/bin/env python
import copy
import torch
import os
import time
import warnings
import numpy as np
from torch.nn.functional import dropout
import torchvision

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverdyn import FedDyn


from flcore.trainmodel.models import *
from utils.mem_utils import MemReporter
from options import args_parser
warnings.simplefilter("ignore")
torch.manual_seed(0)

# vocab_size = 98635
max_len=200
hidden_dim=32

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if model_str == "mlr":
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif args.dataset == "cifar10" or args.dataset == "cifar100":
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(12, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn":
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset == "cifar10" or args.dataset == "cifar100":
                # args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
                args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif args.dataset[:13] == "Tiny-imagenet" or args.dataset[:8] == "Imagenet":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)


        elif model_str == "dnn": # non-convex
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = DNN(1*28*28, 50, num_classes=args.num_classes).to(args.device)
            elif args.dataset == "cifar10" or args.dataset == "cifar100":
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        else:
            pass
# -----------------------------------------------------------------------#
#               Baseline Algorithms
# -----------------------------------------------------------------------#
        # select algorithm
        if args.algorithm == "FedAvg": 
            server = FedAvg(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)
            
        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)
            
        else:
            raise NotImplementedError
        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()
    args = args_parser()
    args.algorithm = "FedProx"  ## FedAvg, FedProx and FedDyn
    args.dataset = "fmnist"
    args.model = "cnn"
    args.num_classes = 10
    args.batch_size = 128
    args.dp_sigma=1
    args.privacy = 'TimeSenFedPLDP'   ## DP-SGD, TimeSenFedPLDP, and No-DP
    args.global_rounds = 50
    print(os.getcwd())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    # print_par(args)
    run(args)
    print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
