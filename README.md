# Introduction
Federated Learning (FL) was conceived as a secure form of distributed learning by keeping private training data local and only communicating public model gradients between clients. However, a slew of gradient leakage attacks proposed to date undermine this claim by proving its insecurity. A common limitation of these attacks is the necessity for extensive auxiliary information, such as model weights, optimizers, and certain hyperparameters (e.g., learning rate), which are challenging to acquire in practical scenarios. Furthermore, several existing algorithms, including FedAvg, circumvent the transmission of model gradients in FL by instead sending model weights, but the potential security breaches of this approach are seldom considered. In this paper, we propose two innovative frameworks, DLM and DLM+, that reveal the potential leakage of private local data of clients when transmitting model weights under the FL framework. We also conduct a series of experiments to elucidate the impact and universality of our attack frameworks. Additionally, we propose and evaluate two defenses against the proposed attacks, assessing their protective efficacy.


## How to Run



## Citation

If this code is useful in your research, you are encouraged to cite our academic paper:
```
@article{zhao2022deep,
  title={Deep leakage from model in federated learning},
  author={Zhao, Zihao and Luo, Mengen and Ding, Wenbo},
  journal={arXiv preprint arXiv:2206.04887},
  year={2022}
}
```
