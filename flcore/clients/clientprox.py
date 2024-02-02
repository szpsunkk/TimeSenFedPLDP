import torch
import numpy as np
import time
import copy
import torch.nn as nn
from utils.privacy import *
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client


class clientprox(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.cou = 0
        self.sigma_log =[]

        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = PerturbedGradientDescent(
            # self.model.parameters(), lr=self.learning_rate, mu=self.mu)
        self.optimizer =torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        if self.privacy == "DP-SGD" or self.privacy == "TimeSenFedPLDP":
            check_dp(self.model)
            initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma, self.epsilon)
            self.epsilon_local = []
            self.delta = []
            self.alpha = []

    def train(self, global_model, cou, data_poison=False,model_poison=False):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        iteration = len(trainloader.dataset)
        max_local_steps = self.local_steps
        if self.privacy == 'TimeSenFedPLDP':
            clip, sigma = calculate_clip_sigma(self.epsilon, self.sample_rate, 60000, iteration, cou, self.decay_rate_mu, self.decay_rate_mu_flag, self.decay_rate_sens, self.line)
            self.optimizer.privacy_engine.set_unit_sigma(sigma)
            self.sigma_log.append(sigma)
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                if model_poison == True: # 模型投毒
                    # print(self.model.state_dict())
                    w = self.sign_attack(self.model.state_dict())
                    self.model.load_state_dict(w)
                    
                if data_poison == True: # 数据投毒
                    y = (y+1)%10
                self.optimizer.zero_grad()
                output = self.model(x)
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = self.loss(output, y) + (self.mu / 2) * proximal_term
                # loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return cou

    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()
            
            
    def sign_attack(self, w):
        w = self.model.state_dict()
        w_avg = copy.deepcopy(w)
        for key in w_avg.keys():
            w_avg[key] = -w[key] * self.model_poison_scale   # 模型里面随便赋值
        return w_avg
