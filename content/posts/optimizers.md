---
title: "Optimizer matters the most!"
date: 2020-04-15
tags: ['optimization', 'machine learning', 'deep learning']
categories: ['optimization', 'deep learning', 'machine learning']
---
# Introduction

As a researcher, most time of my job is to build an appropriate AI prototype for specific tasks. To achieve a satisfactory result, an expected large amount of work i.e tuning hyper-parameters, balancing data, augmentation etc are needed. The most deterministic component of deep learning practice is choosing the appropriate optimization algorithms, which directly affect the training speed and the final predictive performance. To date, there is no theory that adequately explains how to make this choice.

**Adam** and **SGD** optimizers ave played a key role in the literature of deep learning, and they were treated as a breakthrough to optimize a **large volume** of data based **non-convex cases**. However, which one performs better is still debatable. Adam has shown its advantage i.e., surprising fast converging, while SGD and its extension SGD + momentum are proved to yield a better or sometimes way better performance, i.e., higher accuracy on new data. There is a ton of studies of new optimizers by taking advantages of both optimizers. Here, in this blog, I will present varieties of optimizers.

## AdaGrad

The momentum method relies on exponentially weighted moving average to make the update direction of independent variables more consistent, thereby reducing the possibility of divergence. However, it is not $\color{red}{flexible}$ enough for optimization. A simple example is illustrated. At learning rate $\eta$, variables $x_1$ and $x_2$ are updated by 

\begin{align*}
x_{1} \leftarrow x_{1}-\eta \frac{\partial f}{\partial x_{1}}, \quad x_{2} \leftarrow x_{2}-\eta \frac{\partial f}{\partial x_{2}}
\end{align*}


When the gradient values ​​of $x_1$ and $x_2$ are significantly different, e.g., $\eta \frac{\partial f}{\partial x_{1}} - \eta \frac{\partial f}{\partial x_{2}}$, you need to choose a learning rate small enough so that the independent variable does not diverge in the dimension with a large gradient value, but this will cause the independent variable to iterate in the dimension with a small gradient value slow. 

**AdaGrad** is therefore introduced since it adjusts the learning rate in each dimension according to the size of the gradient value of the independent variable in each dimension. 

\begin{align*}
n_{t}=n_{t-1}+g_{t}^{2} \newline
\Delta \theta_{t}=-\frac{\eta}{\sqrt{n_{t}+\epsilon}} * g_{t}
\end{align*}

Essentially, AdaGrad behaves like a regularizer on gradients $g_t$ by multiplying $\frac{1}{\sqrt{n_{t}+\epsilon}}$ where $\epsilon$ is used to guarantee the denominator is non-zero.

### Features:

- At early stage, $g_t$ is small, regularizer is large, they can enlarge gradients;
- At later stage, $g_t$ is large, regularizer is small, they can shorten gradients;
- Appropriate to deal with $\color{blue}{\text{sparse gradients}}$. 

### Drawbacks:

- From formula given above, manual setting a global learning rate is necessary.
- If $eta$ is too large, regularizer will be very sensitive, so as the adjustment on gradient. 
- In the middle and late stages, the accumulation of the gradient squares on the denominator will become larger and larger, e.g., $g_t \rightarrow 0$, making the training end early.
 

## AdaDelta

**AdaDelta** is an extension of AdaGrad. The initial solution is still to adaptively constrain the learning rate, but it is simplified in calculation. AdaGrad accumulates all the squared gradients before update, **while AdaDelta only accumulates fixed-size items, and does not directly store these items, just approximate the corresponding average value.**

\begin{align*}
n_{t}=\nu * n_{t-1}+(1-\nu) * g_{t}^{2} \newline
\Delta \theta_{t}=-\frac{\eta}{\sqrt{n_{t}+\epsilon}} * g_{t}
\end{align*}

Here, AdaDelta still depends on the global learning rate. To avoid that, the author has done a series of changes to approximate second order Newton's method

\begin{align*}
E\left|g^{2}\right|_{t} =\rho * E\left|g^{2}\right|_{t-1}+(1-\rho) * g_{t}^{2} \newline
\Delta \theta_t =-\frac{\sqrt{\sum_{r=1}^{t-1} \Delta \theta_{r}}}{\sqrt{E\left|g^{2}\right|_{t}+\epsilon}}
\end{align*}

### Features

- AdaDelta no longer depends on manual setting global learning rate.
- Very effective acceleration in the early and middle training.
- Later in training, jitter repeatedly around the local minimum.

### Special Case

When $\rho=0.5$, the expectation term $E\left|g^{2}\right|_{t}$ becomes the average of the squared gradients. By taking square root,

\begin{align*}
RMS|g|_{t}=\sqrt{E\left|g^{2}\right|_{t}+\epsilon} \newline
\theta_t = -\frac{\eta}{RMS|g|_{t}} * g_t
\end{align*}

The special case of AdaDelta is named as **RMSprop**, and it is suitable for handling non-stationary targets, which works well for Recurrent Neural Network (RNN).

**Note** that Adam is essentially RMSprop with momentums, and it uses first-order moment estimation and second-order moment estimation to dynamically adjust the learning rate of each parameter. The main advantage of Adam is that after offset correction, each iteration learning rate has a certain range, making the parameters relatively stable.


## AdamW

One of the reason why Adam sometimes perform worse than SGD + momentum in generalization is that L2 regularization is not performing effectively in Adam than SGD. To mention that particularly, 

1. L2 regularization is $\color{red}{\text{not}}$ equal to weight decay in self-adapt learning. Only in stand SGD case, L2 regularization can be treated as weight decay. In self-adapt learning methods, e.g., Adam, L2 regularization leads to smaller shrinkage in weights than weight decay method.
2. Adam optimization with L2 regularization may not be effective. Due to the accumulation of the subtraction term divided by the square of the gradient in the Adam calculation step, the subtraction term with a larger gradient is smaller, so that the weight with a large gradient will not be regularized as the decoupling weight decay. This leads to the inequality of L2 and decoupling weight attenuation regularization of the adaptive gradient algorithm.

![](/post_imgs/adamw.jpeg)


## LazyAdam

Unlike computer vision and other fields, NLP tasks have limited words sampled by each batch, and the gradient estimation of embedding is sparse. For momentum based optimizers, they will overfit embedding easily since it uses the current momentum to update all words even if these words have not been sampled in the dozens of steps. 

LazyAdam is a variant of Adam, and it merges the **sparse** and **dense** Adam optimizers. It only updates moving-average accumulators for sparse variable indices that appear in the current batch, rather than updating the accumulators for all indices. The implementation in Pytorch is as below


````python
import math
import torch
from torch.optim.optimizer import Optimizer

class LazyAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(LazyAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    self.sparse_step(group, p, grad)
                else:
                    self.dense_step(group, p, grad)                

        return loss


    def sparse_step(self, group, param, grad):
        state = self.state[param]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(param.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(param.data)

        state['step'] += 1

        grad = grad.coalesce()  # the update is non-linear so indices must be unique
        grad_indices = grad._indices()
        grad_values = grad._values()
        size = grad.size()

        def make_sparse(values):
            constructor = grad.new
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, size)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        # Decay the first and second moment running average coefficient
        #      old <- b * old + (1 - b) * new
        # <==> old += (1 - b) * (new - old)
        old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        exp_avg.add_(make_sparse(exp_avg_update_values))
        old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
        exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
        exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

        # Dense addition again is intended, avoiding another sparse_mask
        numer = exp_avg_update_values.add_(old_exp_avg_values)
        exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
        denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
        del exp_avg_update_values, exp_avg_sq_update_values

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        param.data.add_(make_sparse(-step_size * numer.div_(denom)))

    
    def dense_step(self, group, param, grad):
        state = self.state[param]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(param.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(param.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        step_size = group['lr'] / bias_correction1

        param.data.addcdiv_(-step_size, exp_avg, denom)
````

### Features

- More efficient way to handle sparse update.
- Avoids overfiting in some degree.
- Large improvements in model training throughput for some applications

### Drawbacks

- Its semantics is slightly different from Adam, so it may lead to different empirical output.
- Might be slower than Adam method.


## AdaBound

Self-adaptive training methods have very unstable output at later stage. In other words, the learning rate is particularly large in some dimensions, and the learning rate in some dimensions is particularly small. AdaBound applies dynamic clipping over learning rate, and it limits the learning rate between $\eta_{l}$ and $\eta_{u}$. 


![](/post_imgs/adabound.png)

It is easy to find that SGD and Adam are special cases for AdaBound. SGD can be viewed as $\eta_l = \eta_u = \alpha^*$, and Adam is $\eta_l =0$ and $\eta_u = \inf$. Due to this setting, AdaBound behaves more like Adam (fast) in the early stage, and it acts like SGD (better convergence) at later stage.

### Features

- Not sensitive to hyper parameters, so save a lot of time for training.
- More flexible boundary functions.
- Smooth transformation from Adam to SGD instead of hard transformation.


## AdaFactor

State-of-the-art pre-training methods, such as Bert, GPT2, or T5, are quite large. Albert applied decomposition to save lot of parameters, however, the calculation of gradients still requires a lot of resources. That is also why a lot smaller model doest mean a lot faster. Adam update process is given as below

\begin{aligned}&g_t = \nabla_{\theta} L(\theta_t) \newline
&m_t = \beta_1 m_{t-1} + \left(1 - \beta_1\right) g_t \newline 
&v_t = \beta_2 v_{t-1} + \left(1 - \beta_2\right) g_t^2 \newline
&\hat{m}_t = m_t\left/\left(1 - \beta_1^t\right)\right. \newline
&\hat{v}_t = v_t\left/\left(1 - \beta_2^t\right)\right. \newline
&\theta_t = \theta_{t-1} - \alpha_t \hat{m}_t\left/\sqrt{\hat{v}_t + \epsilon}\right. 
\end{aligned}

It costs less memory by dropping momentums in Adam,

\begin{aligned}&g_t = \nabla_{\theta} L(\theta_t) \newline
&v_t = \beta_2 v_{t-1} + \left(1 - \beta_2\right) g_t^2 \newline
&\hat{v}_t = v_t\left/\left(1 - \beta_2^t\right)\right. \newline
&\theta_t = \theta_{t-1} - \alpha_t \hat{m}_t\left/\sqrt{\hat{v}_t + \epsilon}\right. 
\end{aligned}

To further reduce the memory cost, low rank decomposition helps approximation of $\hat{v}_t$. Also, new decaying strategy is applied as below

\begin{equation}\hat{\beta}_{2,t} =1 - \frac{1}{t^c}\label{eq:beta2}\end{equation}.

Eventually, AdaFactor is formulated then 


\begin{aligned}
&g_{i, j, t}=\nabla_{\theta} L\left(\theta_{i, j ; t}\right) \newline
&\hat{\beta}_{2, t}=1-t^{-c} \newline
&v_{i ; t}^{(r)}=\hat{\beta}_{2, t} v_{t-1 ; i}^{(r)}+\left(1-\hat{\beta}_{2, t}\right) \sum_{j}\left(g_{i, j ; t}^{2}+\epsilon_{1}\right) \newline
&v_{j, t}^{(c)}=\hat{\beta}_{2, t} v_{t-1 ; j}^{(c)}+\left(1-\hat{\beta}_{2, t}\right) \sum_{i}\left(g_{i, j, t}^{2}+\epsilon_{1}\right) \newline
&\hat{v}_{i j, t}=v_{i, t}^{(r)} v_{j, t}^{(c)} / \sum_{j} v_{j, t}^{(c)} \newline
&u_{t}=g_{t} / \sqrt{\hat{v}_{t}} \newline
&\hat{u}_{t}=u_{t} / \max \left(1, R M S\left(u_{t}\right) / d\right) \times \max \left(\epsilon_{2}, R M S\left(\theta_{t-1}\right)\right)\newline
&\boldsymbol{\theta}_{t}=\boldsymbol{\theta}_{t-1}-\boldsymbol{\alpha}_{t} \hat{\boldsymbol{u}}_{t}
\end{aligned}

The default parameters are

\begin{array}{c|c} 
\epsilon_1 & 10^{-30} \newline
\epsilon_2 & 10^{-3} \newline 
d & 1 \newline
\hat{\beta}_{2,t} & 1 - t^{-0.8} \newline
\end{array}

When parameters is a one dimensional vector instead of a matrix, $\hat{v}^t$ is then updated by 

\begin{align*}
\hat{v}_t = \hat{\beta}_{2,t} v_{t-1} + \left(1 - \hat{\beta}_{2,t}\right) \left(g_t^2+\epsilon_1\right).
\end{align*}

Also, if no learning is defined, the default learning rate will be $a_t = \min\left(10^{-2},\frac{1}{\sqrt{t}}\right)$


### Features

- Faster training
- Less memory cost in GPU
- Better convergence 


## Lamb

When Bert came out, the recorded training time is about 3 days with a cluster of TPUs. LAMB is then introduced to reduce the training time from 3 days to 76 minutes.

![](/post_imgs/lamb.png)

Normally, large batch training will have following problems

- result in the loss of test accuracy, so you need to adjust the hyper-parameters, such as the learning rate. Thus, it is necessary to increase the learning rate linearly or square root as the batch size increases; 
- large learning rate will cause unstable training at beginning, warm-up is needed;
- generalization gap problem: large batch training models will tend to converge to a sharp local minimum point, which will cause training to easily achieve a higher training accuracy, but it is difficult to obtain better test accuracy.


### Features

- Stabilize large batch training 
- Fast 


## LookAhead

LookAhead is found different from other optimizers since it is orthogonal to other optimizers, which means that lookahead can be used to enhance other optimizers. Lookahead optimizer maintains two set of weights, **slow weights** and **fast weights**. Lookahead first updates $k$ iterations of fast weights in the inner loop, and then update slow weights in the direction of the last weights. It is why called lookahead. When Lookahead oscillates in the direction of high curvature, fast weights update quickly advances in the direction of low curvature, and slow weights smoothes the oscillation through parameter interpolation. **The combination of fast weights and slow weights improves learning in the direction of high curvature, reduces variance, and makes Lookahead achieve faster convergence in practice.**


![](/post_imgs/lookahead.png)

### Features

- **Lower variance**
- Robust to hyper-parameters


## Ranger

RAdam (did not introduce here) can be said to be the best foundation that optimizers establish at the beginning of training (CV aspect). RAdam uses a dynamic rectifier to adjust Adam's adaptive momentum according to the variance, and effectively provides automatic warm-up, customized according to the current data set to ensure a solid training start. LookAhead is inspired by the latest advances in the loss of surface understanding of deep neural networks and provides breakthroughs in robust and stable exploration throughout the training period. The combination of RAdam and LookAhead in one optimizer is called Ranger.	

The newest Ranger includes **gradient centralization**, which can be viewed as a projected gradient descent method with a constrained loss function. Gradient centralization is not only able to regularize the weight space and output feature space for improving generalization, but also provide efficient training by improving lipschitzness of loss function and its gradients.

### Features

- Include warm-up for training
- Robust to hyper-parameters



## References

[1] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul), 2121-2159.

[2] Zeiler, M. D. (2012). ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.

[3] Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 4(2), 26-31.

[4] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

[5] Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. "On the Variance of the Adaptive Learning Rate and Beyond." arXiv preprint arXiv:1908.03265 (2019).

[6] Loshchilov, Ilya and Frank Hutter. “Decoupled Weight Decay Regularization.” ICLR (2019).

[7] Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba. Lookahead Optimizer: k steps forward, 1 step back. [Arxiv]

[8] Leslie N. Smith, Nicholay Topin Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates

[9] Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh Large Batch Optimization for Deep Learning: Training BERT in 76 minutes

[10] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson Averaging Weights Leads to Wider Optima and Better Generalization



