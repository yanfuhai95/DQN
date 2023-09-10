# Deep Q-learning Network

## Description

Implementation of Deep Q-learning Network based on [Pytorch](https://www.pytorch.org/).

## How to run

We provide experiments for training model in environments of [gymnasium](https://gymnasium.farama.org/).

In the case of `CartPole`, you can train the model with below steps.

### 1. Clone the repository

```shell
git clone https://github.com/yanfuhai95/DQN.git
```

### 2. Install the DQN module

```shell
cd DQN
pip install -r requirements.txt
pip install -e .
```

### 3. Change to the directory `dqn/experiments/carpole`

```shell
cd cd dqn/experiments/cartpole
```

### 4. Train the model

```shell
python3 train.py
```

After training finished, a model would be saved at the relative path 'model/cart_pole.pth'

### 5. Evaluate the model

```shell
python3 evaluate.py
```

`evaluate.py` would load the model save at 'model/cart_pole.pth'.
