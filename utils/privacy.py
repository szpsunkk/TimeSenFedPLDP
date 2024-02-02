from opacus import PrivacyEngine
from opacus.dp_model_inspector import DPModelInspector
import math
from utils.GaussianCalibrator import calibrateAnalyticGaussianMechanism
import numpy as np

MAX_GRAD_NORM = 0.01
# EPSILON = 1
DELTA = 1 / 60000
EPOCHS = 200
N_ACCUMULATION_STEPS = 1
max_per_sample_grad_norm = 1.0

def initialize_dp(model, optimizer, sample_rate, dp_sigma, epsilon):
    sigma = calculate_sigma_g(epsilon, DELTA)
    # sigma = calculate_sigma_balle(epsilon, DELTA, sample_rate, iteration)
    privacy_engine = PrivacyEngine(
        model,
        sample_rate = sample_rate * N_ACCUMULATION_STEPS,
        epochs = EPOCHS,
        target_epsilon = epsilon,
        target_delta = DELTA,
        noise_multiplier = sigma, 
        max_grad_norm = MAX_GRAD_NORM,
    )
    privacy_engine.attach(optimizer)


def get_dp_params(optimizer):
    return optimizer.privacy_engine.get_privacy_spent(DELTA), DELTA  # 给定这个值利用Moments Accountant遍历α，计算最佳的ϵ


def check_dp(model):
    inspector = DPModelInspector()  # check the model to support DP
    inspector.validate(model)


def dp_step(optimizer, i, len_train_loader):
    # take a real optimizer step after N_VIRTUAL_STEP steps t
    if ((i + 1) % N_ACCUMULATION_STEPS == 0) or ((i + 1) == len_train_loader):
        optimizer.step()
    else:
        optimizer.virtual_step() # take a virtual step

# def dynamic_dp():
def calculate_sigma_g(epsilon, delta, C = 1):  # 常数
    return math.sqrt(2*math.log(1.25/delta)) / (epsilon * C/ 1)

def calculate_sigma_g_dy_l(epsilon, delta, T, alpha = 2.0):  # 线性
    return alpha * math.sqrt(2*math.log(1.25/delta)) / (epsilon * T / 1)

def calculate_sigma_g_dy_e(epsilon, delta, T, alpha = 1.0): # 指数性
    return alpha * math.sqrt(2*math.log(1.25/delta)) / (epsilon * math.exp(T) / 1)

def calculate_sigma_g_dy_p(epsilon, delta, T, alpha = 1.0): # 平方
    return alpha * math.sqrt(2*math.log(1.25/delta)) / (epsilon * math.pow(T, 2) / 1)

def calculate_sigma_g_dy_log(epsilon, delta, T, alpha = 1.0): # 对数
    return alpha * math.sqrt(2*math.log(1.25/delta)) / (epsilon * math.log(T+1) / 1)

def calculate_sigma_balle(epsilon, delta, sampling_rate, iteration):
    
    mu = 1/calibrateAnalyticGaussianMechanism(epsilon = epsilon, delta  = delta, GS = 1, tol = 1.e-12)
    mu_t = math.sqrt(math.log(mu**2/(sampling_rate**2*iteration)+1))
    sigma = 1/mu_t
    return sigma
    

def calculate_clip_sigma(epsilon, sampling_rate, num_data, iteration, step, decay_rate_mu, decay_rate_mu_flag, decay_rate_sens, line):
    # lr = 0.01
    # epochs = 100
    # num_data = 60000
    # batch_size = 128
    # sampling_rate = batch_size/num_data
    # iteration = int(epochs/sampling_rate)
    
    if step==0:
        delta = 1.0/num_data
        sigma_g = calculate_sigma_g(epsilon, DELTA)
        unit_sigma = sigma_g
        clip = max_per_sample_grad_norm * (decay_rate_sens)**step
    else:  
        # delta = 1.0/num_data
        # mu = 1/calibrateAnalyticGaussianMechanism(epsilon = epsilon, delta  = delta, GS = 1, tol = 1.e-12)
        # mu_t = math.sqrt(math.log(mu**2/(sampling_rate**2*step)+1))
        # sigma = 1/mu_t
        if line == 'Line':  # 时间线性
            unit_sigma = calculate_sigma_g_dy_l(epsilon, DELTA, step)
        elif line == 'Exp':  # 指数
            unit_sigma = calculate_sigma_g_dy_e(epsilon, DELTA, step)
        elif line == 'Qua': # 平方
            unit_sigma = calculate_sigma_g_dy_p(epsilon, DELTA, step)
        elif line == 'Log':  # 对数
            unit_sigma = calculate_sigma_g_dy_log(epsilon, DELTA, step)
        elif line == 'Con':  # 常数
            unit_sigma = calculate_sigma_g(epsilon, delta)
    #    print("sigma: {}".format(sigma))
        clip = max_per_sample_grad_norm * (decay_rate_sens)**step

        # if decay_rate_mu_flag:
        #     mu_0 = mu0_search(mu, iteration,decay_rate_mu,sampling_rate,mu_t=mu_t)
        #     unit_sigma = 1/(mu_0/(decay_rate_mu**(step)))
        # else:
        #     unit_sigma = sigma
    return clip, unit_sigma


def cal_step_decay_rate(target_percent,T):
    return np.exp(np.log(target_percent)/T)

def mu0_search(mu,T,decay_rate,p,mu_t=None):
    low_mu = 0.1*mu
    if mu_t:
        high_mu = mu_t
    else:
        high_mu = 50*mu
    for i in range(1000):
        mus = []
        mu_0 = (low_mu+high_mu)/2
        for t in range(T):
            mus.append(mu_0/decay_rate**(t))
        if(p*math.sqrt(sum(np.exp(np.array(mus)**2)-1))>mu):
            high_mu = mu_0
        else:
            low_mu = mu_0
    return mu_0
