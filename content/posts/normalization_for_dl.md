---
title: "Self-adapting techniques: normalization"
date: 2019-09-23
tags: ['deep learning']
categories: ['deep learning']
---

## Introduction 

$\color{blue}{\text{Batch Normalization (BN)}}$ has been treated as one of the standard "plug-in" tool to deep neural networks since its first release. It is been proved to be very helpful in a tons of machine learning applications due to its several advantages as followings:

1. **faster training**
2. **higher learning rate**
3. **easier initialization**
4. **more activations support**
5. **deeper but simpler architecture**
6. **regularization**


**Algorithms**: 


\begin{align*}
	&{\text { Input: Values of } x \text { over a mini-batch: } \mathcal{B}= \{x_{1 \ldots m}\}} \newline
	&{\text { Output: } \{y_{i}=\mathrm{B} \mathrm{N}_{\gamma, \beta} (x_{i})\}} \newline
	&{\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_{i} \qquad \text { // min-batch mean}} \newline
	&{\sigma_{\mathcal{B}}^{2} \leftarrow \frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{\mathcal{B}}\right)^{2} \qquad \text { // mini-batch variance }} \newline
	&{\hat{x}_{i} \leftarrow \frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} \qquad \text { // normalize }} \newline 
	&{y_{i} \leftarrow \gamma \widehat{x}_{i}+\beta \equiv \mathrm{B} \mathrm{N}_{\gamma, \beta}\left(x_{i}\right) \qquad  \text { // scale and shift }}
\end{align*}


Here, $\gamma$ and $\beta$ are the parameters to be learned, the parameters update is based on chain rule like followings:

\begin{align*} 
	\frac{\partial \ell}{\partial \widehat{x}_{i}} &=\frac{\partial \ell}{\partial y_{i}} \cdot \gamma \newline
	\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}} &=\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot\left(x_{i}-\mu_{\mathcal{B}}\right) \cdot \frac{-1}{2}\left(\sigma_{\mathcal{B}}^{2}+\epsilon\right)^{-3 / 2} \newline 
	\frac{\partial \ell}{\partial \mu_{\mathcal{B}}} &=\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \frac{-1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} \newline
	\frac{\partial \ell}{\partial x_{i}} &=\frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \frac{1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}+\frac{\partial \ell}
	{\partial \sigma_{\mathcal{B}}^{2}} \cdot \frac{2\left(x_{i}-\mu_{\mathcal{B}}\right)}{m}+\frac{\partial \ell}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m} \newline
	\frac{\partial \ell}{\partial \gamma} &=\sum_{i=1}^{m} \frac{\partial \ell}{\partial y_{i}} \cdot \widehat{x}_{i} \newline
	\frac{\partial \ell}{\partial \beta} &=\sum_{i=1}^{m} \frac{\partial \ell}{\partial y_{i}} 
\end{align*}


Basically, a short summary for BN is that it is designed to solve $\color{blue}{\text{gradient vanish/explosion}}$ by scaling (normalizing) the $\color{red}{\text{net activation}}$ (output before activation) to values with mean equal to $0$ and variance $1$. Hence, it stabilizes the distribution at each layer. Additionally, It increases the **model capacity** in some degree. Intuitively speaking, BN is naturally applied before activation function (think about $RELU(x) = max(0, x)$) for having scaled distribution of the output. **Note** that it is also applicable to put BN after activation to achieve better performance according to some empirical studies.


Certainly, BN is not a universal solution due to its drawbacks from its nature,

1. when batch size is small, the performance is significantly worse;
2. for some fine-grain tasks, BN will bring negative effects;
3. it is not designed for "dynamic" network, e.g., sequence model;
4. statistics are different between training stage and inference stage.


Therefore, the variety of BN $\[1\]$ has been proposed, e.g., $\color{blue}{\text{layer normalization}} \[2\]$ , $\color{blue}{\text{group normalization}} \[4\]$, $\color{blue}{\text{weight normalization}} \[5\]$ , $\color{blue}{\text{instance normalization}} \[3\]$, and $\color{blue}{\text{PowerNorm}} \[6\]$.

![normalization](/post_imgs/normalization.png)


## Layer Normalization 

Layer normalization (LN) is very commonly to be used in NLP applications. For the design of BERT, both encoders and decoders have applied transformers, which is a block that LN is applied after a multi-head attention mechanism. As I just summarized, BN is not designed for dynamic networks such as RNN, transformers since each batch has different size (text length) and some are really small (like drawback 1 above). Certainly, if large batch is allowed, we can still apply $\color{blue}{\text{bucket sampling}}$ to sort the input texts based on its length, and then apply BN as well. A more natural solution is to apply normalization on layers instead of batches. 


In RNN setting, LN can be applied to each time point, and we can guarantee the statistics is summarized over all $H$ hidden nodes for different time point. For node at time $t$, given the input that hidden state $h_{t-1}$ and input at time t, $x_t$, the output before LN is calculated as 

\begin{align*}
	a^t=W_{h h} h^{t-1}+W_{x h} x^t
\end{align*}


and then we can apply LN on hidden state as below

\begin{align*}
	h^t = \frac{g}{\sqrt{(\sigma^t)^2+\epsilon}} \odot (a^t-\mu^t)+ b  \qquad \mu^{t}=\frac{1}{H} \sum_{i=1}^{H} a_{i}^{t} \qquad \sigma^{t}=\sqrt{\frac{1}{H} \sum_{i=1}^{H}\left(a_{i}^{t}-\mu^{t}\right)^{2}}
\end{align*}


where $g$ is the gain, and $b$ is the bias.


### Note:

The main difference in implementing both normalization methods is that BN takes same feature from different samples (batch), while LN takes different features from the same sample. It is also why sometimes BN's performance is superior than LN due to the fact that the same feature after normalization will remain the original information.


## Instance Normalization

Instance normalization (IN) is proposed to scale the distribution into a even smaller area. By looking at the formula, 

\begin{align*}
	y_{t i j k}=\frac{x_{t i j k}-\mu_{t i}}{\sqrt{\sigma_{t i}^{2}+\epsilon}} \qquad \mu_{t i}=\frac{1}{H W} \sum_{l=1}^{W} \sum_{m=1}^{H} x_{t i l m} \qquad \sigma_{t i}^{2}=\frac{1}{H W} \sum_{l=1}^{W} \sum_{m=1}^{H}\left(x_{t i l m}-m u_{t i}\right)^{2}
\end{align*}


the main difference between BN and IN is that BN normalizes the data across the batch and spatial locations (CNN based), while IN normalizes each batch independently. In other words, BN computes one mean and standard deviation from batch, while IN computes $T$ of them not jointly. 

It is not hard to find out that the beauty of IN is that it is used for **small batch** and **fine-grain** cases since it does not calculate across channels and batches, which will include random noise.


## Group Normalization

A method proposed by Kaimin takes the advantage from both BN and IN, which is group normalization (GN). For example, the batch input data, e.g., image can be described into 4 dimensions, $[N, C, H, W]$, where $N$ is the batch size, $C$ is the channel, and $H$, $W$ are the feature shape, i.e., hight and weight. **What GN does is that it firstly divide channels into groups, and then normalize on groups.** In symbolized version, the original shape $[N, C, H, W]$ will be reshaped by group normalization to $[N, G, C//G, H, W]$, where $G$ stands for group.

The idea behind GN is pretty intuitive. According to CNN method, the extracted feature after filters have the property, invariance. In other words, the features learned from the same data has the same distribution, so the same features can be put in to the same group, which can be understood such as HOG or GIST feature which are group representing features with physical meanings. GN handles cases when batch size matters, and it essentially does the same thing as BN but in group perspective. 

Pytorch code is given 


```python
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


def group_norm(input, group, running_mean, running_var, weight=None, bias=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5):
    r"""Applies Group Normalization for channels in the same group in each data sample in a
    batch.
    See :class:`~torch.nn.GroupNorm1d`, :class:`~torch.nn.GroupNorm2d`,
    :class:`~torch.nn.GroupNorm3d` for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')

    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _instance_norm(input, group, running_mean=None, running_var=None, weight=None,
                       bias=None, use_input_stats=None, momentum=None, eps=None):
        # Repeat stored stats and affine transform params if necessary
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)

        #norm_shape = [1, b * c / group, group]
        #print(norm_shape)
        # Apply instance norm
        input_reshaped = input.contiguous().view(1, int(b * c/group), group, *input.size()[2:])

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, weight=weight, bias=bias,
            training=use_input_stats, momentum=momentum, eps=eps)

        # Reshape back
        if running_mean is not None:
            running_mean_orig.copy_(running_mean.view(b, int(c/group)).mean(0, keepdim=False))
        if running_var is not None:
            running_var_orig.copy_(running_var.view(b, int(c/group)).mean(0, keepdim=False))

        return out.view(b, c, *input.size()[2:])
    return _instance_norm(input, group, running_mean=running_mean,
                          running_var=running_var, weight=weight, bias=bias,
                          use_input_stats=use_input_stats, momentum=momentum,
                          eps=eps)

class _GroupNorm(_BatchNorm):
    def __init__(self, num_features, num_groups=1, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=False):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(_GroupNorm, self).__init__(int(num_features/num_groups), eps,
                                         momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return group_norm(
            input, self.num_groups, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

class GroupNorm2d(_GroupNorm):
    r"""Applies Group Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    https://arxiv.org/pdf/1803.08494.pdf
    `Group Normalization`_ .
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        num_groups:
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples:
        >>> # Without Learnable Parameters
        >>> m = GroupNorm2d(100, 4)
        >>> # With Learnable Parameters
        >>> m = GroupNorm2d(100, 4, affine=True)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class GroupNorm3d(_GroupNorm):
    """
        Assume the data format is (B, C, D, H, W)
    """
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
```


## Weight Normalization

An alternative method to BN and LN is the weight normalization (WN). BN and LN normalize data, while WN normlize on weights. The idea of WN is to decompose weights vector $w$ into a parameter vector $v$ and a parameter scalar $g$, the math form is 


\begin{align*}
w = \frac{g}{\|v\|} \ v
\end{align*}


where $\|v\|$ is the norm of $v$. When $v = w$ and $g = ||w||$, WN will remain the original way of calculation, so WN can increase the capacity of networks. $v$ and $g$ can be update by SGD as


$$
\nabla_{g} L=\frac{\nabla_{\mathbf{w}} L \cdot \mathbf{v}}{\|\mathbf{v}\|} \quad \nabla_{\mathbf{v}} L=\frac{g}{\|\mathbf{v}\|} \nabla_{\mathbf{w}} L-\frac{g \nabla_{g} L}{\|\mathbf{v}\|^{2}} \mathbf{v}
$$


where $L$ is the loss function, and $ \nabla_{\mathbf{w}} L$ is the gradient of $w$ under $L$, the above SGD process can be also written in geometry asepect


\begin{align*}
\nabla_{\mathbf{v}} L=\frac{g}{\|\mathbf{v}\|} M_{\mathbf{w}} \nabla_{\mathbf{w}} L \quad \text { with } \quad M_{\mathbf{w}}=I-\frac{\mathbf{w} \mathbf{w}^{\prime}}{\|\mathbf{w}\|^{2}}
\end{align*}


and the formula deriving process is 


\begin{array}{c}{\nabla_{\mathbf{v}} L=\frac{g}{\|\mathbf{v}\|} \nabla_{\mathbf{w}} L-\frac{g \nabla_{g} L}{\|\mathbf{v}\|^{2}} \mathbf{v}} \\ {=\frac{g}{\|\mathbf{v}\|} \nabla_{\mathbf{w}} L-\frac{g}{\|\mathbf{v}\|^{2}} \frac{\nabla_{\mathbf{w}} L \cdot \mathbf{v}}{\|\mathbf{v}\|} \mathbf{v}} \\ {=\frac{g}{\|\mathbf{v}\|}\left(I-\frac{\mathbf{v} \mathbf{v}^{\prime}}{\|\mathbf{v}\|^{2}}\right) \nabla_{\mathbf{w}} L} \\ {=\frac{g}{\|\mathbf{v}\|}\left(I-\frac{\mathbf{w} \mathbf{w}^{\prime}}{\|\mathbf{w}\|^{2}}\right) \nabla_{\mathbf{w}} L} \\ {=\frac{g}{\|\mathbf{v}\|} M_{\mathbf{w}} \nabla_{\mathbf{w}} L}\end{array}


Two key parts are reflected from above process, 

1. WN will scale down weights' gradients by $\frac{g}{\|v\|}$;
2. WN will project gradients into a direction far away from $\nabla_w L$.

Therefore, they faster model convergence.

### Equivalence to BN

When neural network only has one layer and the batch follows independent mean 0 and variance 1 distribution, WN is equivalent to BN.

### Initialization

The method of intializing WN is different from BN, and it is suggested from the original paper, 

- $v$ follows normal distirbution with mean 0 and standard deviation 0.05
- $g$ and $b$ leverage statistics based on first batch for initialization
	
	
$$g \leftarrow \frac{1}{\sigma[t]} \quad b \leftarrow -\frac{\mu[t]}{\sigma[t]}$$
	

An interesting finding from WN is to use $\color{blue}{\text{mean-only BN}}$, so it only applies mean reduction but not dividing variance. The reason behind is that the original BN (divide by variance) will include extra noise. Some work also shows that WN + mean-only BN will yield better generalization than BN but way slower convergence.

A quick summary of WN advantages is:

- faster convergence
- more robustness 
- applicable to dynamic networks, RNN
- not sensitive to noise, can be used in GAN and RL


## PowerNorm

Batch Normalization (BN) is widely adopted in CC, but it leads to significant performance degradation when naively used in NLP. Instead, Layer Normalization (LN) is the standard normalization scheme used in NLP, especially transformers based models. It is still not clear why BN performs worse and LN works better. Research $\[7\]$ presents the idea of "Internal Covariate Shift" was viewed as incorrect/incomplete. In particular, the recent study of $\[8\]$ argued that the underlying reason that BN helps training is that it results in a smoother loss landscape, and it was confirmed in $\[9\]$. 

Author illustrates what will happen after replacing BN with LN in transformers as below

![](https://pic4.zhimg.com/80/v2-30cf484e6c4e4ffe498daf52c4935a8f_1440w.jpg)

In the above picture, blue is the result of ResNet20's image classification in Cifar-10, and orange is the result of Transformer + BN's translation in IWSLT14. The X-axis is the training time, and the Y-axis is the Euclidean distance based on the statistical value of the batch and its corresponding moving average. It can be seen that the fluctuation of ResNet20 on the Cifar-10 task is very small, while the Transformer with BN not only oscillates violently, but also has extreme outliers, which will lead to inaccurate estimates of $\mu$ and $\sigma$. Hence, generalization decreases due to inconsistency among training and testing. What an interesting findings from the results!

BN forces the data following a normal distribution with a mean of 0 and a variance of 1. However, in the case where the mean variance of the data itself oscillates violently, forcing the moving average will have a bad effect. Therefore, the author proposes a new scale method, only forcing the data to have **unit quadratic mean**, PN-V:

\begin{array}{c}
\psi_{B}^{2}=\frac{1}{B} \sum x_{i}^{2} \newline
\hat{X}=\frac{X}{\psi_{B}} \newline
Y=\gamma \cdot \hat{X}+\beta
\end{array}

Now, only one statistic is used in forward pass, and backward pass requires only $g_{\psi}^2$ to update,

\begin{align*}
\frac{\partial \mathcal{L}}{\partial x_{i}}=\frac{1}{\psi_{B} \gamma} \frac{\partial \mathcal{L}}{\partial y_{i}}-\frac{1}{\psi_{B} \gamma B} \sum_{i \in B}\left(\frac{\partial \mathcal{L}}{\partial y_{j}} \hat{x}_{i} \hat{x_{j}}\right)
\end{align*}

and the oscillate is significantly reduced

![](https://pic2.zhimg.com/80/v2-3acbbfe411b07ac3fd15cf8f804286dd_1440w.jpg)

As the plot shown above, it is easy to find that there are still some outliers. To help that, Author suggested moving average to calculate $\psi$, 

\begin{align*}
\hat{X}_{t}=\frac{X_{t}}{\psi_{t-1}} \newline
Y_{t}=\gamma \cdot \hat{X}_{t}+\beta \newline
\psi_{t}^{2}=\alpha \psi_{t-1}^{2}+(1-\alpha) \psi_{B}^{2}
\end{align*}


## Other Extensions

- $\color{blue}{\text{Switchable Normalization \[10\]}}$, proposed by ***SenseTime Research***. 
![](https://raw.githubusercontent.com/switchablenorms/Switchable-Normalization/master/teaser.png)Chinese version, click [here](https://zhuanlan.zhihu.com/p/39296570?utm_source=wechat_session&utm_medium=social&utm_oi=70591319113728).
- For **small batch** and **non-i.i.d. data**, e.g., image segmentation, $\color{blue}{\text{EvalNorm \[11\]}}$ corrects normalization statistics to improve performances.
- $\color{blue}{\text{Moving Average Batch Normalization (MABN) \[12\]}}$ replaces batch statistics from small batch with moving averages.


## References

$\[1\]$: Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

$\[2\]$: Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

$\[3\]$: Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempit- sky. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022, 2016.  

$\[4\]$: Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European Conference on Computer Vision (ECCV), pages 3–19, 2018.

$\[5\]$: Salimans, T., Kingma, D.~P. (2016) Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. arXiv e-prints arXiv:1602.07868.

$\[6\]$: Shen, S., Yao, Z., Gholami, A., Mahoney, M., Keutzer, K. (2020) Rethinking Batch Normalization in Transformers. arXiv e-prints arXiv:2003.07845.

$\[7\]$: Sergey Ioffe and Christian Szegedy. Batch nor- malization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

$\[8\]$: Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How does batch normalization help optimization? In Advances in Neural Information Processing Systems, pages 2483–2493, 2018.

$\[9\]$: Zhewei Yao, Amir Gholami, Kurt Keutzer, and Michael W Mahoney. PyHessian: Neural networks through the lens of the Hessian. arXiv preprint arXiv:1912.07145, 2019.

$\[10\]$: Luo, P., Ren, J., Peng, Z., Zhang, R., Li, J. (2018) Differentiable Learning-to-Normalize via Switchable Normalization. arXiv e-prints arXiv:1806.10779.

$\[11\]$: Saurabh Singh and Abhinav Shrivastava. EvalNorm: Estimating batch normalization statistics for evaluation. In Proceedings of the IEEE International Conference on Computer Vision, pages 3633–3641, 2019.

$\[12\]$: Junjie Yan, Ruosi Wan, Xiangyu Zhang, Wei Zhang, Yichen Wei, and Jian Sun. Towards stabilizing batch statistics in backward propagation of batch normalization. arXiv preprint arXiv:2001.06838, 2020.


