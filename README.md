# TimeSenFedPLDP
In the FL system, we proposed a time-sensitive PLDP-based FL (TimeSenFLDP) mechanism to achieve different privacy levels of the model of vehicles over sharing time steps.



## Algorithm
a time-sensitive personalized local differential privacy-based Federated Learning algorithm
![image](https://github.com/szpsunkk/TimeSenFedPLDP/assets/47681094/7b8238d8-bca3-47be-9395-196bac923052)

## Deployment

### Install the environment
``
pip install -r requirements.txt
``
The experiment is based on Pytorch and [Opacus](https://opacus.ai/). After installing the Opacus, we need to modify the file ``PrivacyEngine``
