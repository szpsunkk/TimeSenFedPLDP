import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
import copy
import torch.nn.functional as F
  
class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        self.cou = 0
        self.sigma_log =[]

        # differential privacy
        if self.privacy == "DP-SGD" or self.privacy == "TimeSenFedPLDP":
            check_dp(self.model)
            initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma, self.epsilon)
            self.epsilon_local = []
            self.delta = []
            self.alpha = []

    def train(self, cou, data_poison=False,model_poison=False):
        trainloader = self.load_train_data()
        
        start_time = time.time()
        # self.model.to(self.device)
        self.model.train()
        # sampling_rate = self.batch_size/60000
        iteration = len(trainloader.dataset)
        max_local_steps = self.local_steps
        if self.privacy == 'TimeSenFedPLDP':
            clip, sigma = calculate_clip_sigma(self.epsilon, self.sample_rate, 60000, iteration, cou, self.decay_rate_mu, self.decay_rate_mu_flag, self.decay_rate_sens, self.line)
            self.optimizer.privacy_engine.set_unit_sigma(sigma)
            self.sigma_log.append(sigma)
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                
                if model_poison == True: # 模型投毒
                    # print(self.model.state_dict())
                    w = self.sign_attack(self.model.state_dict())
                    self.model.load_state_dict(w)
                    
                if data_poison == True: # 数据投毒
                    y = (y+1)%10
                output = self.model(x)
                # self.evaluate(i)
                
                loss = self.loss(output, y)
                loss.backward()

                if self.privacy == "DP-SGD":
                    dp_step(self.optimizer, i, len(trainloader))
                else:
                    self.optimizer.step()
                # self.cou +=1
                
            
        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return cou
            
    def sign_attack(self, w):
        w = self.model.state_dict()
        w_avg = copy.deepcopy(w)
        for key in w_avg.keys():
            w_avg[key] = -w[key] * self.model_poison_scale   # 模型里面随便赋值
        return w_avg
    
    def evaluate(self, i):
            
        if i % 10 == 0:
            print("---------------------------------------------")
            print("Evaluate the local model at step {}".format(i)) 
            stats = self.test_metrics()  # test_acc 0, test_num 1, auc 2, test_acc_black 3, test_num_black 4, test_acc_white 5, test_num_white 6
            test_acc = stats[0]*1.0 / stats[1]
            test_auc = stats[2]
            # accs = [a / n for a, n in zip(stats[2], stats[1])]
            # aucs = [a / n for a, n in zip(stats[3], stats[1])]
            if self.dataset == 'adult' or self.dataset == 'bank' or self.dataset == 'german':
                
                test_acc_black = stats[3] / stats[4]
                test_acc_white = stats[5] / stats[6]
            print("Averaged Test Accurancy: {:.4f}".format(test_acc * 100))
            print("Averaged Test AUC: {:.4f}".format(test_auc))
            # self.print_(test_acc, train_acc, train_loss)
            # print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            # print("Std Test AUC: {:.4f}".format(np.std(aucs)))
            
            print("black people Test accuracy: {}".format(test_acc_black * 100))
            print("white people Test accuracy: {}".format(test_acc_white * 100))
        

        
