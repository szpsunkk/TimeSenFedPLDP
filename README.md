# TimeSenFedPLDP
In the FL system, we proposed a time-sensitive PLDP-based FL (TimeSenFLDP) mechanism to achieve different privacy levels of the model of vehicles over sharing time steps.
Paper`Personalized Privacy-Preserving Distributed Artificial Intelligence for Digital-Twin-Driven Vehicle Road Cooperation`


## Algorithm
a time-sensitive personalized local differential privacy-based Federated Learning algorithm
![image](https://github.com/szpsunkk/TimeSenFedPLDP/assets/47681094/7b8238d8-bca3-47be-9395-196bac923052)

## Deployment

### Install the environment
```
pip install -r requirements.txt
```
The experiment is based on Pytorch and [Opacus](https://opacus.ai/). After installing the Opacus, we need to modify the file ``opacus.PrivacyEngine``(Opacus version 0.15.0), and add the following code to the end of the file ``opacus.PrivacyEngine``.
```
    def set_clip(self,new_clip):
        self.max_grad_norm = new_clip
        self.clipper.norm_clipper.flat_value = new_clip

    def set_unit_sigma(self,unit_sigma):
        self.noise_multiplier = unit_sigma
```
### Create the dataset
We use the Mnist and Fmnist to create the iid and noniid datasets for federated learning clients.
```
cd dataset
python Mnist.py noniid balance pat  # the noniid Mnist dataset for clients
or
python Mnist.py iid balance pat     # the iid Mnist dataset for clients
```
Then you will get the following result:
```
Client 0         Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 1         Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 2         Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 3         Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 4         Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 5         Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 6         Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 7         Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 8         Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 9         Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 10        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 11        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 12        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 13        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 14        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 15        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 16        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 17        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 18        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 19        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 20        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 21        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 22        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 23        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 24        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 25        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 26        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 27        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 28        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 29        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 30        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 31        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 32        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 33        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 34        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 35        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 36        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 37        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 38        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 39        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 40        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 41        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 42        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 43        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 44        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 45        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 46        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 47        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 48        Size of data: 1395      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 138), (1, 157), (2, 139), (3, 142), (4, 136), (5, 126), (6, 137), (7, 145), (8, 136), (9, 139)]
--------------------------------------------------
Client 49        Size of data: 1645      Labels:  [0 1 2 3 4 5 6 7 8 9]
                 Samples of labels:  [(0, 141), (1, 184), (2, 179), (3, 183), (4, 160), (5, 139), (6, 163), (7, 188), (8, 161), (9, 147)]
--------------------------------------------------
Total number of samples: 70000
The number of train samples: [1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1046, 1233]
The number of test samples: [349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 
349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 412]

Saving to disk.

Finish generating dataset.
```
And run the ``mian.py`` file:
```
python mian.py 
```
or 
```
python main.py --dataset mnist --global_rounds 2 --local_steps 5 --algorithm FedProx  --model cnn --privacy DP-SGD --epsilon 1
```
The parameter meaning:
```
--dataset: the local training dataset (mnist, fmnist)
--global_rounds: the total global rounds in the Federated Learning training process
--local_steps: the local steps in clients
--algorithm: the federated learning algorithm, including FedAvg, FedProx and FedDyn
--model: the training model,including cnn, dnn
--privacy: the privacy preservation mechanism, including NO-privacy, DP-SGD and TimeSenFedPLDP
--epsilin: the parameter of differential privacy, [0,10]

```

### The results:
#### The test accuracy of iid and noniid settings with mnist and fmnist dataset
![image](https://github.com/szpsunkk/TimeSenFedPLDP/assets/47681094/f27a6ab3-9c6d-4ccc-8424-33aa562c47d8)
#### The test accuracy of different privacy budgets:
![image](https://github.com/szpsunkk/TimeSenFedPLDP/assets/47681094/407ba3ea-e261-4b5e-89b3-427aba07c7b7)


