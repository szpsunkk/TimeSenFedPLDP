#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse   

def args_parser():  
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=2)
    parser.add_argument('-m', "--model", type=str, default="dnn")
    parser.add_argument('-p', "--predictor", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=128)
    parser.add_argument('-lbse', "--batch_size_end", type=int, default=3000)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.001,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=2)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-ar', "--attack_ratio", type=float, default=1,
                        help="Ratio of attacked clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    # FedProx parameter
    parser.add_argument('-mu', "--mu", type=float, default=0.02,
                        help="Proximal rate for FedProx")
    # FedDyn parameter
    parser.add_argument('-al', "--alpha", type=float, default=0.05)
    # privacy parameter
    parser.add_argument('-dp', "--privacy", type=str, default="DP-SGD",
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=5)
    parser.add_argument('-eps', "--epsilon", type=float, default=0.1)
    parser.add_argument('-dpmu', "--decay_rate_mu", type=float, default=0.8)
    parser.add_argument('-dpsens', "--decay_rate_sens", type=float, default=0.5)
    parser.add_argument('-dpmuf', "--decay_rate_mu_flag", type=float, default=True)
    parser.add_argument('-dpsensf', "--decay_rate_sens_flag", type=float, default=True)
    parser.add_argument('-li', "--line", type=str, default='Line', help="the method adding privacy: line: T, exp: exp(T) and pow: T^2")
    # 
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')
    # attack
    parser.add_argument('-dpn','--data_poison', type=str2bool, default=False,
                        help='True: data poisoning attack, False: no attack')
    parser.add_argument('-mp','--model_poison', type=str2bool, default=False,
                        help='True: model poisoning attack, False: no attack')
    parser.add_argument('-mps','--model_poison_scale', type=float, default=0.1,
                        help='scale of model poisoning attack (0.1 or 10)')
    parser.add_argument('--num_commondata', type=float, default=1000,
                    help='number of public data which server has') 
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.5,
                        help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    args = parser.parse_args()
    
    return args

def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
