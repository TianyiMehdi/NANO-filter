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
$$
\begin{equation}
\nonumber
\begin{aligned}
x_{t+1} &= \begin{bmatrix}
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} x_t+\xi_t,
\\
y_t &= \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix} x_t+\zeta_t .
\end{aligned}
\end{equation}
$$
### Test:
```bash
cd experiments
# 1. For system without outlier
python wiener/wiener_NANO.py
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
$$
\begin{aligned}
x_{t+1} &= \begin{bmatrix}
1 & \frac{\sin \omega_t \Delta t}{\omega_t} & 0 & -\frac{1-\cos \omega_t \Delta t}{\Omega_t} & 0 \\
0 & \cos \omega_t \Delta t & 0 & -\sin \omega_t \Delta t & 0 \\
0 & \frac{1-\cos \omega_t \Delta t}{\omega_t} & 1 & \frac{\sin \omega_t \Delta t}{\omega_t} & 0 \\
0 & \sin \omega_t \Delta t & 0 & \cos \omega_t \Delta t & 0 \\
0 & 0 & 0 & 0 & 1
\end{bmatrix} x_t+\xi_t, \\
y_t &= \begin{bmatrix}
\sqrt{p_{x,t}^2+p_{y,t}^2+h^2} \\
\mathrm{atan}\left(\frac{p_{y,t}}{p_{x,t}}\right) \\
\mathrm{atan}\left(\frac{h}{\sqrt{p_{x,t}^2+p_{y,t}^2}}\right) \\
\frac{p_{x,t} \dot{p}_{x,t}+p_{y,t} \dot{p}_{y,t}}{\sqrt{p_{x,t}^2+p_{y,t}^2+h^2}}
\end{bmatrix}+\zeta_t . \\
\end{aligned}
$$

### Test:
```bash
cd experiments
# 1. For system without outlier
python air_traffic/air_traffic_NANO.py
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
$$
\begin{aligned}
    \begin{bmatrix}
        p_{x,t+1}\\
        p_{y,t+1}\\
        \theta_{t+1}
\end{bmatrix}&=\begin{bmatrix}
        p_{x,t}\\
        p_{y,t}\\
        \theta_t
    \end{bmatrix} + \begin{bmatrix}
        v_t \cdot \cos\theta_t\\
        v_t \cdot \sin\theta_t\\
        \omega_t
    \end{bmatrix}\cdot\Delta t + \xi_t, \\
    y_t &= [d^1_t \ d^2_t \ d^3_t \ \alpha^1_t \ \alpha^2_t \ \alpha^3_t]^\top + \zeta_t,
\end{aligned}
$$

### Test:
```bash
cd experiments
python ugv/ugv_NANO.py \
    --loss_type beta_likelihood_loss \
    --beta 2e-2 \
    --n_iterations 1 \
```
### Figure:

