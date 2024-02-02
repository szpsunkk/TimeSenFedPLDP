import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.batch_size_end = args.batch_size_end
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.global_steps = args.global_rounds

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.line = args.line

        # privacy
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.sample_rate = self.batch_size / self.train_samples
        self.epsilon = args.epsilon
        self.decay_rate_mu = args.decay_rate_mu
        self.decay_rate_mu_flag = args.decay_rate_mu_flag
        self.decay_rate_sens = args.decay_rate_sens
        self.decay_rate_sens_flag = args.decay_rate_sens_flag
        
        # ! attack model
        self.adv_client_model = copy.deepcopy(args.model)
        self.adv_client_dataset = args.dataset
        self.data_poison = args.data_poison
        self.model_poison = args.model_poison
        self.model_poison_scale = args.model_poison_scale
        # FedProx
        self.mu = args.mu
        self.equalized_odds_record = []
        self.demographic_parity_record = []
        self.local_acc_log = []
        self.local_auc_log = []
        # Sensitive attribute 



    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def load_test_data_batch(self):
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, self.batch_size_end, drop_last=False, shuffle=True)
    
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device) 
        self.model.eval()   # 主要是进行模型的推断作用 

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        test_acc_black = []
        test_acc_white = []
        
        equalized_odds_log = []
        demographic_parity_log = []
        
        
        with torch.no_grad():
            for x, y in testloaderfull:

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                y_predicted = torch.argmax(output, dim=1)
                
                # y = self.to_categorical(y, 2)
                # if self.dataset == 'adult':
                #     # test_acc += (output.ge(0.5)[:,0] == y).sum().item()
                #     # test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #     test_acc += torch.sum(torch.max(F.softmax(output),1)[1] == y).cpu()
                #     # test_acc_black += torch.sum(torch.max(F.softmax(output),1)[1] * x[ :, 8])
                #     test_num_black_i = torch.sum(x[:,8]==torch.tensor(0))
                #     test_num_black += test_num_black_i
                #     test_num_white += y.shape[0] - test_num_black_i 
                #     aa = (torch.max(F.softmax(output),1)[1] == y)
                #     for i in range(len(aa)):
                #         if aa[i] == torch.tensor(1):
                #             if x[i,8] == torch.tensor(0):
                #                  test_acc_black += 1
                #             else:
                #                  test_acc_white += 1
                    
                # else:
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                if self.dataset == 'adult':
                    y_prob.append(output.detach().cpu().numpy())
                    y_true.append(self.to_categorical(y.detach().cpu().numpy(), self.num_classes))
                    mf = MetricFrame(metrics=accuracy_score, y_true=y.cpu(), y_pred=y_predicted.cpu(), sensitive_features=x[:,self.sen].cpu())
                    test_acc_black.append(mf.by_group[0])
                    test_acc_white.append(mf.by_group[1])  
                    equalized_odds = equalized_odds_difference(y.cpu(), y_predicted.cpu(), sensitive_features=x[:,self.sen].cpu())  
                    demographic_parity = demographic_parity_difference(y.cpu(), y_predicted.cpu(), sensitive_features=x[:,self.sen].cpu())    
                    # 
                    equalized_odds_log.append(equalized_odds)
                    demographic_parity_log.append(demographic_parity)  
                elif  self.dataset == 'bank':
                    y_prob.append(output.detach().cpu().numpy())
                    y_true.append(self.to_categorical(y.detach().cpu().numpy(), self.num_classes))
                    mf = MetricFrame(metrics=accuracy_score, y_true=y.cpu(), y_pred=y_predicted.cpu(), sensitive_features=x[:,self.sen].cpu())
                    test_acc_black.append(mf.by_group[1])
                    test_acc_white.append(mf.by_group[2])  
                    equalized_odds = equalized_odds_difference(y.cpu(), y_predicted.cpu(), sensitive_features=x[:,self.sen].cpu())  
                    demographic_parity = demographic_parity_difference(y.cpu(), y_predicted.cpu(), sensitive_features=x[:,self.sen].cpu())    
                    # 
                    equalized_odds_log.append(equalized_odds)
                    demographic_parity_log.append(demographic_parity)
                elif  self.dataset == 'german':
                    y_prob.append(output.detach().cpu().numpy())
                    y_true.append(self.to_categorical(y.detach().cpu().numpy(), self.num_classes))
                    mf = MetricFrame(metrics=accuracy_score, y_true=y.cpu(), y_pred=y_predicted.cpu(), sensitive_features=x[:,self.sen].cpu())
                    test_acc_black.append(mf.by_group[0])
                    test_acc_white.append(mf.by_group[1])  
                    equalized_odds = equalized_odds_difference(y.cpu(), y_predicted.cpu(), sensitive_features=x[:,self.sen].cpu())  
                    demographic_parity = demographic_parity_difference(y.cpu(), y_predicted.cpu(), sensitive_features=x[:,self.sen].cpu())    
                    # 
                    equalized_odds_log.append(equalized_odds)
                    demographic_parity_log.append(demographic_parity)
                elif self.dataset == 'smart_grid':
                    y_prob.append(y_predicted.detach().cpu().numpy())
                    y_true.append(y.detach().cpu().numpy())
                else:
                    y_prob.append(output.detach().cpu().numpy())
                    y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        # self.model.cpu()
        # self.save_model(self.model, 'model')
        if self.dataset == 'adult' or self.dataset == 'bank' or self.dataset == 'german':
            # print("equalized_odds:{}, demographic_parity:{}".format(np.sum(equalized_odds_log) / len(equalized_odds_log), np.sum(demographic_parity_log) / len(demographic_parity_log))) 
            self.equalized_odds_record.append(np.sum(equalized_odds_log) / len(equalized_odds_log))
            self.demographic_parity_record.append(np.sum(demographic_parity_log) / len(demographic_parity_log))
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # print("black acc {}".format(test_acc_black/test_num_black * 100.0))
        # print("white acc {}".format(test_acc_white/test_num_white * 100.0))
        # if self.dataset == "adult":
        #     auc = metrics.roc_auc_score(y_true, y_prob[:, 0], average='micro')
        # else:
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        if self.dataset == 'adult' or self.dataset == 'bank' or self.dataset == 'german':
            cc = [test_acc, test_num, auc, test_acc_black, test_acc_white, np.sum(equalized_odds_log) / len(equalized_odds_log), np.sum(demographic_parity_log) / len(demographic_parity_log)]
        else:
            cc = [test_acc, test_num, auc]
            self.local_acc_log.append(test_acc / test_num)
        return cc   # 测试精度，测试数量，测试

    def test_metrics_batch(self):
        testloaderfull = self.load_test_data_batch()
        for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                y_predicted = torch.argmax(output, dim=1)
                metrics = {
                            "accuracy": accuracy_score,
                            "precision": precision_score,
                            "false positive rate": false_positive_rate,
                            "false negative rate": false_negative_rate,
                            "selection rate": selection_rate,
                            "count": count,
                        }
                mf = MetricFrame(metrics=metrics, y_true=y.cpu(), y_pred=y_predicted.cpu(), sensitive_features=x[:,8].cpu())
                break
        return mf.by_group

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        loss = 0
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)  #!
            train_num += y.shape[0]
            if self.dataset == 'adult' or self.dataset == 'bank' or self.dataset == 'german':
                # loss += self.loss(output[:,0], y.to(torch.float)).item() * y.shape[0]
                loss += self.loss(output, y)
            else:
                loss += self.loss(output, y).item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return loss, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
    def sign_attack(self, w):
        w_avg = copy.deepcopy(w)
        for key in w_avg.keys():
            w_avg[key] = -w[key] * self.model_poison_scale   # 模型里面随便赋值
        return w_avg
    
    def to_categorical(self, y, num_classes):
        """
        1-hot encodes a tensor 
        """
        return np.eye(num_classes, dtype='uint8')[y]         