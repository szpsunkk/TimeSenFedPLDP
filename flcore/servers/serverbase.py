import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn
from os.path import join as oj
import csv
import pandas as pd
# import norm
from utils.data_utils import read_client_data
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.attack_ratio = args.attack_ratio
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.save_folder_name = args.save_folder_name
        self.model = args.model
        
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.epsilon = args.epsilon

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_aucs = []
        self.rs_test_accs = []
        self.rs_train_loss = []
        
        self.rs_test_acc_black = []
        self.rs_test_acc_white = []
        self.df = []

        self.times = times
        self.eval_gap = args.eval_gap
        # self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate
        
        # # data_poison
        # self.data_poison = args.data_poison
        # # model_poison
        # self.model_poison = args.model_poison
        # self.model_poison_scale = args.model_poison_scale
        # if self.dataset != "compas":
        #     self.train_dateset = read_client_data(self.dataset,self.num_clients-1, is_train=True)  #!! sageflow算法
        
        # 增加
        # self.flag_attack = args.flag_attack
        # self.num_adv_clients = args.adversaries_num
        # # self.adv_client_type = args.adversaries_type
        # self.adv_client_model = copy.deepcopy(args.model)
        # self.adv_client_dataset = args.dataset
        # self.selected_adv_clients = []
        # self.adv_clients = []
        
        # self.gamma = args.gamma
        
        # self.is_fair = args.is_fair # 是否采用公平性算法
        

    def set_clients(self, args, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            
            client = clientObj(args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
    def set_clients_sk(self, args, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # train_data = read_client_data(self.dataset, i, is_train=True)
            # test_data = read_client_data(self.dataset, i, is_train=False)
            
            client = clientObj(args, 
                            id=i, 
                            train_samples=1000, 
                            test_samples=1000, 
                            train_slow=train_slow, 
                            send_slow=send_slow
                            )
            self.clients.append(client)

    def set_attack_clients(self, args, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_adv_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            
            adv_client = clientObj(args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.adv_clients.append(adv_client)
        
    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        selected_clients = list(np.random.choice(self.clients, self.join_clients, replace=False))
        self.selected_clients = selected_clients
        return selected_clients

    def select_attack_clients(self):
        select_attack_clients = list(np.random.choice(self.clients, self.attack_clients , replace=False))
        self.selected_adv_clients = select_attack_clients
        return select_attack_clients

    def send_models(self):
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            client.set_parameters(self.global_model)
            
            
    def send_adv_models(self):
        assert (len(self.selected_clients) > 0)

        for adv_client in self.adv_clients:
            adv_client.set_parameters(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
        # if self.flag_attack == True:
        #     tot_samples = 0
        #     for adv_client in self.select_attack_clients:
        #         self.uploaded_weights.append(adv_client.train_samples)
        #         tot_samples += adv_client.train_samples
        #         self.uploaded_ids.append(adv_client.id)  #TODO
        #         self.uploaded_models.append(adv_client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0]) # 第一个模型
        for param in self.global_model.parameters():  # 这个模型参数为零
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
            

    def add_gloable_gradient_updates(self, grad_update_2, weight = 1.0):
        # assert len(self.global_model.parameters()) == len(
        #     grad_update_2), "Lengths of the two grad_updates not equal"
        
        for param_1, param_2 in zip(self.global_model.parameters(), grad_update_2):
            param_1.data += param_2.data * weight
    
    def add_update_to_model(self, model, update, weight=1.0, device=None):
        """
        模型更新梯度，权重weight
        """
        if not update: return model
        if device:
            model = model.to(device)
            update = [param.to(device) for param in update]
                
        for param_model, param_update in zip(model.parameters(), update):  # 每一层的梯度相加
            param_model.data += weight * param_update.data
        return model

    def flatten_gloable_model(self):
	    
        return torch.cat([update.data.view(-1) for update in self.global_model.parameters()])
    
    def flatten(self, grad_update):
        return torch.cat([update.data.view(-1) for update in grad_update])
    
    def normlize(self, flattened):
        norm_value = norm(flattened) + 1e-7
 
    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
            # print(type(server_param.data), server_param.size())
        # for name,param in self.global_model.named_parameters():
        #     # print(name, param)
        #     if name == "fc1.0.weight":
        #         self.global_model.state_dict()["fc1.0.weight"].data += client_model.state_dict()["fc1.0.weight"].data.clone() * w
        #     elif name == "fc1.0.bias":
        #         self.global_model.state_dict()["fc1.0.bias"].data += client_model.state_dict()["fc1.0.bias"].data.clone() * w
        #     elif name == "fc2.0.weight":
        #         self.global_model.state_dict()["fc2.0.weight"].data += client_model.state_dict()["fc2.0.weight"].data.clone() * w
        #     elif name == "fc2.0.bias":
        #         self.global_model.state_dict()["fc2.0.bias"].data += client_model.state_dict()["fc2.0.bias"].data.clone() * w
                
            # print(server_param[0].size())
            # print(server_param[0])
            

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        participant_str = 'algo({})-N({})-A({})-dp({})-mp({})-model_poison_sacle({})-join_ratio({})_privacy({})_dp_sigma({})_Local_e({})_local_step({})_joint_ratio_({})_lr_({})_line({})'.format(self.algorithm, self.num_clients, self.attack_clients, self.data_poison, self.model_poison, self.model_poison_scale, self.join_ratio, self.privacy, self.dp_sigma, self.epsilon, self.local_steps, self.join_ratio, self.learning_rate, self.line)
        folder = oj(
            "./results",
            self.dataset,
            self.algorithm, 
            participant_str
        )
        if not os.path.exists(folder):
            os.makedirs(folder)

        # if (len(self.rs_test_acc)):
        algo = algo + "_" + self.goal + "_" + str(self.times)
        if self.algorithm == "FedMFG":
            file_path = "{}_gamma{}.h5".format(algo, self.gamma)
            file_path_txt = "txt_{}_gamma{}.txt".format(algo, self.gamma)
            file_config = "config_{}_gamma{}.txt".format(algo, self.gamma)
            flie_acc = "acc_{}_gamma{}.csv".format(algo, self.gamma)
            flie_remove_num="remove_num_{}_gamma{}.txt".format(algo, self.gamma)
            flie_rs="rs_{}_gamma{}.csv".format(algo, self.gamma)
            with open(folder + "/" + flie_remove_num, "w", encoding='utf-8', newline='') as f:
                f.write("attack_client_id:" + "\n\n")
                # f.write(str(self.attack_client_id))
                f.write("remove_num:"+"\n\n")
                # f.write(str(self.remove_num))
            # past_rs = torch.stack(self.past_rs).detach().cpu().numpy()
            df = pd.DataFrame(self.past_rs)
            df.to_csv(oj(folder, flie_rs), index=False) 
        # elif self.algorithm == "FedProx":
        #     file_path = "{}_mu{}.h5".format(algo, self.mu)
        #     file_path_txt = "txt_{}_mu{}.txt".format(algo, self.mu)
        #     file_config = "config_{}_mu{}.txt".format(algo, self.mu)
        #     flie_acc = "acc_{}_mu{}.csv".format(algo, self.mu)
        elif self.algorithm == "FedAvg_sk":
            if not os.path.exists(folder+ "/" + "exp{}-exf{}-fair{}-privacy{}-algorithm{}/".format(self.exp, self.exf, self.fair, self.privacy_sk, self.fair_algorithm)):
                os.makedirs(folder+ "/" + "exp{}-exf{}-fair{}-privacy{}-algorithm{}/".format(self.exp, self.exf, self.fair, self.privacy_sk, self.fair_algorithm))
            with open(folder + "/" + "exp{}-exf{}-fair{}-privacy{}-algorithm{}/".format(self.exp, self.exf, self.fair, self.privacy_sk, self.fair_algorithm) + "fedavg_s.txt", "w", encoding='utf-8', newline='') as f:
                f.write("error:" + "\n")
                for item in self.error:
                    f.write(str(item) + "\n")
                f.write("disc:" + "\n") 
                for item in self.disc: 
                    f.write(str(item) + "\n")                
                f.write("metric:" + "\n\n")
                f.write(str(self.metric) + "\n\n")
            for client in self.selected_clients:
                with open(folder + "/" + "exp{}-exf{}-fair{}-privacy{}-algorithm{}/".format(self.exp, self.exf, self.fair, self.privacy_sk, self.fair_algorithm) + "fedavg_c{}.txt".format(client.id), "w", encoding='utf-8', newline='') as f:
                    f.write("error:" + "\n")
                    for item in client.error:
                        f.write(str(item) + "\n")
                    f.write("disc:" + "\n")
                    for item in client.disc:
                        f.write(str(item) + "\n")                
                    f.write("metric:" + "\n\n")
                    f.write(str(client.metric) + "\n\n")
        else:
            file_path = "{}.h5".format(algo)
            file_path_txt = "txt_{}.txt".format(algo)
            file_config = "config_{}.txt".format(algo)
            flie_acc = "acc_{}.csv".format(algo)

        # with h5py.File(result_path + "/" + file_path, 'w') as hf:
        #     hf.create_dataset('rs_test_acc', data=self.rs_test_acc*100)
        #     hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
        #     hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            
            with open(folder + "/" + file_path_txt, "w", encoding='utf-8', newline='') as f:
                f.write("rs_test_acc:" + "\n\n")
                f.write(str(self.rs_test_acc*100))
                f.write("rs_test_auc:"+"\n\n")
                f.write(str(self.rs_test_auc))
                f.write("rs_train_loss:"+"\n\n")
                f.write(str(self.rs_train_loss))
                f.write("auc std:"+"\n\n")
                f.write(str(self.rs_test_aucs))
                f.write("acc std:"+"\n\n")
                f.write(str(self.rs_test_accs))
            
            with open(folder + "/" + file_config, "w", encoding='utf-8', newline='') as f:
                f.write("model:"+"\n\n")
                f.write(str(self.model))
                f.write("dataset"+"\n\n")
                f.write(str(self.dataset))
            for i in range(self.num_clients):
                    if self.privacy == "normal" or self.privacy == 'dynamic' or self.privacy == 'time':
                        f = open(folder + "/" + "eps_{}.csv".format(i), "w", encoding='utf-8', newline='')
                        wr = csv.writer(f)
                        for j in range(len(self.selected_clients[i].epsilon_local)):
                            wr.writerow([j + 1, self.selected_clients[i].epsilon_local[j]])
                            
                        f = open(folder + "/" + "alpha_{}.csv".format(i), "w", encoding='utf-8', newline='')
                        wr = csv.writer(f)
                        for j in range(len(self.selected_clients[i].alpha)):
                            wr.writerow([j + 1, self.selected_clients[i].alpha[j]])
                        
                        f = open(folder + "/" + "test_acc_{}.csv".format(i), "w", encoding='utf-8', newline='')
                        wr = csv.writer(f)
                        for j in range(len(self.selected_clients[i].local_acc_log)):
                            wr.writerow([j + 1, self.selected_clients[i].local_acc_log[j]])
                # if self.privacy == "time":
                #     f = open(folder + "/" + "sigma_{}.csv".format(i), "w", encoding='utf-8', newline='')
                #     wr = csv.writer(f)
                #     for j in range(len(self.selected_clients[i].sigma_log)):
                #         wr.writerow([j + 1, self.selected_clients[i].sigma_log[j]])
        
        # if self.dataset == 'adult' or self.dataset == 'bank' or self.dataset == 'german':
        #     file_acc_black = "acc_black_{}_dp_{}.csv".format(algo, self.dp_sigma)
        #     file_acc_white = "acc_white_{}_dp_{}.csv".format(algo, self.dp_sigma)
        #     file_loss = "loss_{}_dp_{}.csv".format(algo, self.dp_sigma)
        #     # 
        #     file_privacy = "privacy_{}_dp_{}.csv".format(algo, self.dp_sigma)
        #     f = open(folder + "/" + flie_acc, "w", encoding='utf-8', newline='')
        #     wr = csv.writer(f)
        #     for i in range((len(self.rs_test_acc))):
        #         wr.writerow([i + 1, self.rs_test_acc[i] * 100])  
            
        #     f = open(folder + "/" + file_acc_black, "w", encoding='utf-8', newline='')
        #     wr = csv.writer(f)
        #     for i in range((len(self.rs_test_acc_black))):
        #         wr.writerow([i + 1, self.rs_test_acc_black[i] * 100])
                
        #     f = open(folder + "/" + file_acc_white, "w", encoding='utf-8', newline='')
        #     wr = csv.writer(f)
        #     for i in range((len(self.rs_test_acc_white))):
        #         wr.writerow([i + 1, self.rs_test_acc_white[i] * 100])

        #     f = open(folder + "/" + file_loss, "w", encoding='utf-8', newline='')
        #     wr = csv.writer(f)
        #     for i in range((len(self.rs_train_loss))):
        #         wr.writerow([i + 1, self.rs_train_loss[i] * 100])  
                
        #     for i in range(self.num_clients):
        #         plt.figure(figsize=(13,7))

        #         plt.subplot(231)
        #         x=['white','black']
        #         plt.bar(x,self.df[i]["accuracy"], hatch='-')
        #         plt.title('accuracy')

        #         plt.subplot(232)
        #         plt.bar(x,self.df[i]["precision"], hatch='/')
        #         plt.title('precision')

        #         plt.subplot(233)
        #         plt.bar(x,self.df[i]["false positive rate"], hatch='//')
        #         plt.title('false positive rate')

        #         plt.subplot(234)
        #         plt.bar(x,self.df[i]["false negative rate"], hatch='/+')
        #         plt.title('false negative rate')

        #         plt.subplot(235)
        #         plt.bar(x,self.df[i]["selection rate"], hatch='//+')
        #         plt.title('selection rate')

        #         plt.subplot(236)
        #         plt.bar(x,self.df[i]["count"], hatch='//-')
        #         plt.title('count')
                
        #         plt.savefig(folder + "/" + "test_{}.jpg".format(i))
        #         if self.privacy == "normal":
        #             f = open(folder + "/" + "eps_{}.csv".format(i), "w", encoding='utf-8', newline='')
        #             wr = csv.writer(f)
        #             for j in range(len(self.selected_clients[i].epsilon_local)):
        #                 wr.writerow([j + 1, self.selected_clients[i].epsilon_local[j]])
                        
        #             f = open(folder + "/" + "alpha_{}.csv".format(i), "w", encoding='utf-8', newline='')
        #             wr = csv.writer(f)
        #             for j in range(len(self.selected_clients[i].alpha)):
        #                 wr.writerow([j + 1, self.selected_clients[i].alpha[j]])
                        
        #         f = open(folder + "/" + "equalized_odds_{}.csv".format(i), "w", encoding='utf-8', newline='')
        #         wr = csv.writer(f)
        #         for j in range(len(self.selected_clients[i].equalized_odds_record)):
        #             wr.writerow([j + 1, self.selected_clients[i].equalized_odds_record[j]])
                    
        #         f = open(folder + "/" + "demographic_parity_{}.csv".format(i), "w", encoding='utf-8', newline='')
        #         wr = csv.writer(f)
        #         for j in range(len(self.selected_clients[i].demographic_parity_record)):
        #             wr.writerow([j + 1, self.selected_clients[i].demographic_parity_record[j]])
        #     # f = open(folder + "/" + file_privacy, "w", encoding='utf-8', newline='')
        #     # wr = csv.writer(f)
        #     # for i in range((len(self.select_clients[0].epsilon_local))):
        #     #     wr.writerow([i + 1, self.select_clients[0].epsilon_local[i].detach().cpu().numpy() * 100])  
        #     f = open(folder + "/" + "test_{}.txt".format(i), "w", encoding='utf-8', newline='')
        #     wr = csv.writer(f)
        #     for i in range((len(self.df))):
        #         wr.writerow([i + 1, self.df[i]])  
        # else:
        #     f = open(folder + "/" + flie_acc, "w", encoding='utf-8', newline='')
        #     wr = csv.writer(f)
        #     for i in range((len(self.rs_test_acc))):
        #         wr.writerow([i + 1, self.rs_test_acc[i] * 100])
        f.close()
                
        

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_correct_black = []
        tot_correct_white = []
        tot_eo = []
        tot_dp = []
        if self.dataset == 'adult' or self.dataset == 'bank' or self.dataset == 'german':
            for c in self.selected_clients:
                mt = c.test_metrics()  # ct 0, ns 1, auc 2, ct_b 3, ct_w 4, eo 5, dp 6
                tot_correct.append(mt[0]*1.0)
                tot_correct_black.append(sum(mt[3]) / len(mt[3]))
                tot_correct_white.append(sum(mt[4]) / len(mt[4]))
                tot_auc.append(mt[2]*mt[1])
                num_samples.append(mt[1])
                tot_eo.append(mt[5])
                tot_dp.append(mt[6])

        else:
            for c in self.selected_clients:
                me = c.test_metrics() # ct 0 , ns 1, auc 2
                tot_correct.append(me[0]*1.0)
                tot_auc.append(me[2]*me[1])
                num_samples.append(me[1])

        ids = [c.id for c in self.selected_clients]
        if self.dataset == 'adult' or self.dataset == 'bank' or self.dataset == 'german':
            cc = [ids, num_samples, tot_correct, tot_auc, tot_correct_black, tot_correct_white, tot_eo, tot_dp]
        else:
            cc = [ids, num_samples, tot_correct, tot_auc]

        return cc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.selected_clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        
        stats = self.test_metrics()  # ids 0, num_samples 1, tot_acc 2, tot_auc 3, tot_acc_black 4, tot_acc_white 5, fair(tot_eo 6, tot_dp 7)
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        if self.dataset == 'adult':
            
            test_acc_black = sum(stats[4]) / len(stats[4])
            test_acc_white = sum(stats[5]) / len(stats[5])
            
            test_eo = sum(stats[6]) / len(stats[6])
            test_dp = sum(stats[7]) / len(stats[7])
            
            self.rs_test_acc_black.append(test_acc_black)
            self.rs_test_acc_white.append(test_acc_white)
            print("black people Test accuracy: {}".format(test_acc_black * 100))
            print("white people Test accuracy: {}".format(test_acc_white * 100))
            print("EO error: {}".format(test_eo))
            print("DP error: {}".format(test_dp))
        elif self.dataset == 'bank' or self.dataset == 'german':
            male = sum(stats[4]) / len(stats[4])
            femal = sum(stats[5]) / len(stats[5])
            self.rs_test_acc_black.append(male)
            self.rs_test_acc_white.append(femal)
            test_eo = sum(stats[6]) / len(stats[6])
            test_dp = sum(stats[7]) / len(stats[7])
            print("male Test accuracy: {}".format(male * 100))
            print("female Test accuracy: {}".format(femal * 100))
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
            self.rs_test_auc.append(test_auc)
            self.rs_test_accs.append(accs)
            self.rs_test_aucs.append(aucs)
        else:
            acc.append(test_acc)
            
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc * 100))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        
    
    def evaluate_end(self):
        
        for c in self.selected_clients:
            df = c.test_metrics_batch()
            self.df.append(df)
            
        # # testloaderfull = self.load_test_data()
        # self.model.eval()
        # for x, y in testloaderfull:

        #     if type(x) == type([]):
        #         x[0] = x[0].to(self.device)
        #     else:
        #         x = x.to(self.device)
        #     y = y.to(self.device)
        #     output = self.model(x)
        #     y_predicted = torch.argmax(output, dim=1)
        #     mf = MetricFrame(metrics=accuracy_score, y_true=y.cpu(), y_pred=y_predicted.cpu(), sensitive_features=x[:,8].cpu())
        #     metrics = {
        #                 "accuracy": accuracy_score,
        #                 "precision": precision_score,
        #                 "false positive rate": false_positive_rate,
        #                 "false negative rate": false_negative_rate,
        #                 "selection rate": selection_rate,
        #                 "count": count,
        #             }
        #     self.df = mf.by_group.plot.bar(
        #                         subplots=True,
        #                         layout=[3, 3],
        #                         legend=False,
        #                         figsize=[12, 8],
        #                         title="Show all metrics",) 
    
    
    def evaluate_entropy(self, test_dataset):
        """entropy

        Args:
            test_dataset (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.global_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        # device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
        criterion = nn.NLLLoss().to(self.device)
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        self.global_model.to(self.device)
        batch_losses = []
        batch_entropy = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                output, out = self.global_model(images)
                # output = self.global_model(images)

                #Compute the entropy
                Information = F.softmax(out, dim=1) * F.log_softmax(out, dim=1)
                entropy  = -1.0 * Information.sum(dim=1)
                average_entropy = entropy.mean().item()

                batch_loss = criterion(output, labels)
                batch_losses.append(batch_loss.item())

                _, pred_labels = torch.max(output,1)
                pred_labels = pred_labels.view(-1)

                pred_dec = torch.eq(pred_labels, labels)
                current_acc = torch.sum(pred_dec).item() + 1e-8


                batch_entropy.append(average_entropy)

                correct += current_acc
                total += len(labels)


            accuracy  = correct/total

        return accuracy, sum(batch_losses)/len(batch_losses), sum(batch_entropy)/len(batch_entropy)
    
    def inference(self, model, test_dataset):
            criterion = nn.NLLLoss().to(self.device)
            model.eval()
            loss, total, correct = 0.0, 0.0, 0.0
            testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(testloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs,_ = model(images)
                    batch_loss = criterion(outputs, labels)
                    loss += batch_loss.item()

                    _, pred_labels = torch.max(outputs, 1)
                    pred_labels = pred_labels.view(-1)
                    correct += torch.sum(torch.eq(pred_labels, labels)).item()
                    total += len(labels)

                accuracy = correct/total
            return accuracy, loss
        
    def test_inference(args, model, test_dataset):

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
        criterion = nn.NLLLoss().to(device)
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        batch_losses = []
        batch_entropy = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)

                output, out = model(images)

                #Compute the entropy
                Information = F.softmax(out, dim=1) * F.log_softmax(out, dim=1)
                entropy  = -1.0 * Information.sum(dim=1)
                average_entropy = entropy.mean().item()

                batch_loss = criterion(output, labels)
                batch_losses.append(batch_loss.item())

                _, pred_labels = torch.max(output,1)
                pred_labels = pred_labels.view(-1)

                pred_dec = torch.eq(pred_labels, labels)
                current_acc = torch.sum(pred_dec).item() + 1e-8


                batch_entropy.append(average_entropy)

                correct += current_acc
                total += len(labels)


            accuracy  = correct/total

        return accuracy, sum(batch_losses)/len(batch_losses), sum(batch_entropy)/len(batch_entropy)
    
    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True
    
    def cosine_similarity(self, gradient1, gradient2):
    
        return F.cosine_similarity(self.flatten(gradient1), gradient2, 0, 1e-10)
    

