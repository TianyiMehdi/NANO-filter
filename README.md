# Natural Gradient Gaussian Approximation (NANO) Filter

This is the official python code for the paper "Nonlinear Bayesian Filtering with Natural Gradient Gaussian Approximation". You can find the preprint of the paper in [Link](https://arxiv.org/pdf/2410.15832v1). 

Please contact the corresponding author of the code at cwh19@mails.tsinghua.edu.cn or mehdizhang@126.com.

## Index

1. [Installation](#1-installation)
2. [Example: Wiener Velocity Model](#2-example-wiener-velocity-model)
3. [Example: Air-Traffic Control Model](#3-example-air-traffic-control-model)
4. [Example: Unmanned Ground Vehicle Localization](#4-example-unmanned-ground-vehicle-localization)

## 1. Installation

1. Clone the repository.

```bash
git clone https://github.com/TianyiMehdi/NANO-filter.git
```

2. Install the dependencies.

```bash
pip install -r requirements.txt
```

## 2. Example: Wiener Velocity Model
### Model:
![formula](./figures/formulas/wiener.png)

### Test:
```bash
cd experiments
# 1. For system without outlier
python wiener/wiener_NANO.py \
    --measurement_outlier_flag False \
    --n_iterations 1 \
# 2. For system with outlier, and you can try to change the loss_type and loss hyperparameters to see the difference 
python wiener/wiener_NANO.py \
    --measurement_outlier_flag True \
    --loss_type beta_likelihood_loss \
    --beta 9e-4 \
    --n_iterations 1 \
```
### Figure:

## 3. Example: Air-Traffic Control Model
### Model:
![formula](./figures/formulas/air_traffic.png)

### Test:
```bash
cd experiments
# 1. For system without outlier
python air_traffic/air_traffic_NANO.py \
    --measurement_outlier_flag False \
    --n_iterations 1 \
# 2. For system with outlier, and you can try to change the loss_type and loss hyperparameters to see the difference 
python wiener/wiener_NANO.py \
    --measurement_outlier_flag True \
    --loss_type beta_likelihood_loss \
    --beta 2e-2 \
    --n_iterations 1 \
```

### Figure:

## 4. Example: Unmanned Ground Vehicle Localization
### Model:
![formula](./figures/formulas/ugv.png)

### Test:
```bash
cd experiments
python ugv/ugv_NANO.py \
    --loss_type beta_likelihood_loss \
    --beta 2e-2 \
    --n_iterations 1 \
```
### Figure:

