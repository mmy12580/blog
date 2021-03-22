---
title: "Activation: magician of deep learning"
date: 2019-11-23
tags: ['deep learning']
categories: ['deep learning']
---

# Overview:


Activation functions play a crucial rule in neural networks because they are the nonlinearities which have been attributed to the success story of deep learning. At present, the most popular activation functions are **ReLU** and its extended work such as **LReLU**, **PReLu**, **ELU**, **SELU**, and **CReLU** etc. However, none of them is guaranteed to perform better then others in all applications, so it becomes fundamental to understand their advantages and disadvantages in order to achieve better performances in specific applications. This blog will first introduce common types of non-linear activation functions, and then I will introduce which to choose on challenging NLP tasks. 


# Properties

**In general**, activation functions have properties as followings:

1. **non-linearity**: The non-linear activations functions are used not only to stimulate like real brains but also to enhance the ability of representation to approximate the data distribution. In other words, it increases large capacity of model to generalize the data better;
2. **differentiable**: Due to the non-convex optimization problem, deep learning considers back-propagation which is essentially chain rule of derivatives;
3. **monotonic**: Monotonic guarantees single layer is convex;
4. $f(x) \approx x$: When activation function satisfies this property, if values after initialization is small, the training efficiency will increase; if not, initialization needs to be carefully set;
5. **domain**: When the output of activation functions is determined in a range, the gradient based optimization method will be stable. However when the output is unlimited, the training will be more efficient, but choosing learning rate will be necessarily careful.


# Comparison

## Sigmoid

Let us first talk about the classic choice, **sigmod** function, which has formula as 
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$
The name "sigmoid" comes from its shape, which we normally call "S"-shaped curve. 

### Advantages:
* Mapping values to (0, 1) so it wont blow up activation
* Can be used as the output layer to give credible value
* Easy derivatives:
    
\begin{align*}
\sigma'(x) &= - \frac{1}{(1 + e^{-x})^2} (-e^{-x}) \newline
      &= \frac{1}{1 + e^{-x}} \frac{e^{-x}}{1 + e^{-x}} \newline
      &= \frac{1}{1 + e^{-x}} \frac{1 + e^{-x} - 1}{1 + e^{-x}} \newline
      &= \sigma(x)(1 - \sigma(x))
\end{align*}
      

### Disadvantages:
* **Gradient Vanishing**: When $\sigma(x) \rightarrow 0$ or $\sigma(x) \rightarrow 1$, the $\frac{\partial \sigma}{\partial x} \rightarrow 0$. Another intuitive reason is that the $\max f'(x) = 0.25$ when $x=0.5$. That means every time the gradient signal flows through a sigmoid gate, its magnitude always diminishes by one quarter (or more);
* Non-zero centered output: Imagine if x is all positive and all negative, what result will $f'(x)$ has? It sloweres the convergence rate;
* Slow: Exponential computation is relatively slower comparing to ReLu


## Tanh

To solve the non-zero centered output, **tanh** is introduced since its domain is from [-1, 1]. Mathematically, it is just transformed version of sigmoid:

$$ \tanh(x) = 2\sigma(2x -1) = \frac{1 - e^{-2x}}{1 + e^{-2x}} $$

### Advantages:
* $\color{blue}{Zero-centered}$ output: Release the burden of initialization in some degree; Also, it fasters the convergence. 
* Easy derivatives: 


\begin{align*}
\tanh'(x) &= \frac{\partial \tanh}{\partial x} = (\frac{\sin x}{\cos x})' \newline
      &= \frac{\sin'x \cos x + \sin x \cos'x}{\cos^2 x} \newline
      &= \frac{\cos^2 x - sin^2 x}{\cos^2 x} \newline
      &= 1 - \frac{\sin^2 x}{\cos^2 x} = 1 - \tanh^2(x)
\end{align*}
 

### Disadvantages:
* Gradient Vanishing: When $\tanh(x) \rightarrow 1$ or $\tanh(x) \rightarrow -1$, $\tanh'(x) \rightarrow 0$
* Slow: Exponential computation is still included

## ReLU

**ReLU** has become the most popular method in deep learning applications. The idea behind is very simple, 

$$ReLu(x) = \max(0, x)$$

### Advantages:
* Solves gradient vanishing problem
* Faster computation leads to faster convergence
* Even simpler derivative


### Disadvantages:
* Non-zero centered
* **Dead ReLU problem**: Some of the neurons wont be activated. Possible reasons: 1. Unlucky initialization 2. Learning rate is too high. (Small learning rate, Xavier Initialization and Batch Normalization help).

### How does ReLU solves gradient vanish problem?

Assume a DNN with 3 layers and activation function as $S$. Given inputs $x$ and weights $W_i$ and bias $b_i$, the output of third layer is $f_3 =S(W_3 * S(W_2 * S(W_1 * x + b_1)+ b_2)+b_3)$, and the derivative of $W_1$ is 

\begin{aligned}
\frac{\partial L}{\partial W_{1}} &=\frac{\partial L}{\partial f_{3}} \cdot \frac{\partial S\left(W_{3} S\left(W_{2} S\left(W_{1} x+b_{1}\right)+b_{2}\right)+b_{3}\right)}{\partial W_{1}} \newline
&=\frac{\partial L}{\partial f_{3}} \cdot \frac{\partial S}{\partial a_{3}} \cdot \frac{\partial W_{3} S\left(W_{2} S\left(W_{1} x+b_{1}\right)+b_{2}\right)+b_{3}}{\partial W_{1}} \newline
&=\frac{\partial L}{\partial f_{3}} \cdot \frac{\partial S}{\partial a_{3}} \cdot W_{3} \cdot \frac{\partial S\left(W_{2} S\left(W_{1} x+b_{1}\right)+b_{2}\right)}{\partial W_{1}} \newline
&=\cdots \newline
&=\frac{\partial L}{\partial f_{3}} \cdot \frac{\partial S}{\partial a_{3}} \cdot W_{3} \cdot \frac{\partial S}{\partial a_{2}} \cdot W_{2} \cdot \frac{\partial S}{\partial a_{1}} \cdot \frac{\partial a_{1}}{\partial W_{1}}
\end{aligned}

where $a_i$ is the linear combination before activation for layer $i$. As it is shown above, **the partial derivatives $\partial S / \partial a_i$ is the reason of gradient vanish** because 

\begin{align*}
f^{\prime}(x)=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}} \quad \in\left(0, \frac{1}{4}\right]
\end{align*} 

so that $0< \partial S/ \partial a_1≤0.25$，$0 < \partial S/ \partial a_2≤0.25$， $0 < \partial S/\partial a_3 ≤ 0.25$, the product will be 

\begin{align*}
0<\frac{\partial S}{\partial a_{3}} \frac{\partial S}{\partial a_{2}} \frac{\partial S}{\partial a_{1}} \leq 0.015625
\end{align*}

If there are 20 layers, then 

\begin{align*}
0<\frac{\partial S}{\partial a_{20}} \frac{\partial S}{\partial a_{19}} \ldots \frac{\partial S}{\partial a_{1}} \leq 0.25^{20}=9.094 \times 10^{-13}
\end{align*}

**ReLu's derivative is either 0 or 1**, so as long as the derivative on a path in the gradient of the neural network is 1, then no matter how many layers the network has, the gradient of the next few layers of the network can be propagated to the first few layers of the network.

## LeakyReLU

To solve ReLU problems, *LeakyReLU* is proposed to solve dead area and non-zero centers problems. The only changes is that when $x \le 0$, activation is then calculated as $f(x) = a * x$. The derivatives will be either 1 or $a$, and normally $a$ is set to be smaller than 1, e.g., $a = 0.2$.


### Advantages:

- Solves dead area problem by allowing a small value $a$
- Fast since no exponential calculation is included

### Disadvantages:

- Still non-zero centered output
- Cant avoid gradient explosion
- May not converge as fast as ReLU

A quick adjustment on LeaykRelu is to parameterize the value of $a$, so the choice of $a$ can be learned from the data. This method is called PReLU, and it converges fast and have good generalization.


## ELU

What slows down the learning is the bias shift which is present in ReLUs. Those who have mean activation larger than zero and learning causes bias shift for the following layers. **ELU** is designed as an alternative of ReLU to reduce the bias shift by pushing the mean activation toward zero. 


\begin{split}
    ELU(x) &= \alpha (\exp(x) - 1), && \quad x \le 0 \\
           &= x, &&  \quad  x > 0   
\end{split}


### Advantages:
* Zero-Centered outputs
* No Dead ReLU issues
* Seems to be a merged version of LReLU and PReLU

### Disadvantages:
* Slow due to exponential calculation
* Saturates for the large negative values


## SELU

The last common non-linear activation function is **SELU**, scaled exponential linear unit. It has **self-normalizing properties** because the activations that are close to zero mean and unit variance, propagated through network layers, will converge towards zero mean and unit variance. This, in particular, makes the learning highly robust and allows to train networks that have many layers.


\begin{split}
    SELU(x) &= \lambda \alpha (\exp(x) - 1), && \quad x \le 0 \\
           &= \lambda x, &&  \quad  x > 0   
\end{split}


which has gradient as 


\begin{split}
    \frac{\partial d}{\partial x} SELU(x) &= SELU(x) + \lambda \alpha, && \quad x \le 0 \\
           &= \lambda, &&  \quad  x > 0   
\end{split}

where $\alpha = 1.6733$ and $\lambda = 1.0507$.

### Advantages:

- Self-normalizing is faster than external normalization e.g, batch normalization.
- Avoids gradient vanish or explosion (proved in paper by theorem 2 and 3)

 
$\color{blue}{Question}$: Would *SELU*, *ELU* be more useful than **Batch Normalization**?



# Activation functions on advanced NLP

## Swish

Swish is another activation trying to solve ReLU problems, and its defined as 

\begin{align*}
\text{swish}(x)=x\cdot\sigma(x)=\frac{x}{1+e^{-x}}
\end{align*}

The idea of Swish is very similar to GLU, and GLU's formula is given as 

\begin{align*}
(\boldsymbol{W}_1\boldsymbol{x}+\boldsymbol{b}_1)\otimes \sigma(\boldsymbol{W}_2\boldsymbol{x}+\boldsymbol{b}_2)
\end{align*}

GLU is a production of two set of parameters with one of them activated by sigmoid, which performs like a gate to control the output. It is also why named as GLU. Swish can be viewed as those two set of parameters are equal, therefore only one set of parameters needs to be trained. 

### Why it works?

Swish is not saturated near the origin, only the negative area is far away from the origin area is saturated, and ReLu is also saturated in half of the space near the origin (positive area). When we train a model, the initialization parameters generally follow uniform initialization or normal distribution. Hence, the average value is generally 0. That is to say, half of the initialization parameters are in the ReLu saturation region. In the beginning, half of the parameters were not used. Especially due to strategies such as batch normalization, the output automatically approximates the normal distribution with a mean of 0, so in these cases, half of the parameters are located in the ReLu saturation region. In contrast, Swish is a little better, because it also has a certain unsaturated zone on the negative half, so the utilization of the parameters is greater.

## GeLU 

All the NLP benchmarks scores have been updated since BERT was released. Pre-training method becomes the standard way of solving NLP downstream tasks, such as entity recognition, Q&A systems, sentiment analysis, and so on. BERT applied a new activation method, *GELU*, Gaussian error linear unit. Essentially, GELU uses Gaussian distribution to be have like a Bernoulli trial, so that when input $x$ is small, it is more easier to be set to zero. **Isn't this very similar to idea of Swish?**

\begin{align*}
GELU(x) &= x * P(X \le x) \newline  
&=  x * \Phi(x) \newline
&\approx 0.5 x\left(1+\tanh \left(\sqrt{2 / \pi}\left(x+0.044715 x^{3}\right)\right)\right)
\end{align*}

Its derivative is then calculated as

\begin{array}{c}
\mathrm{GELU}^{\prime}(x)=0.5 \tanh \left(0.0356774 x^{3}+0.797885 x\right) \\
+\left(0.0535161 x^{3}+0.398942 x\right) \sec^{2}\left(0.0356774 x^{3}+0.797885 x\right)+0.5
\end{array}

  
In code wise, gelu is performed differently among pre-training methods

````python
# OpenAI version
def openai_gpt_gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# original BERT
def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# more popular choice (default)
def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
````


## Mish

Inspired by *Swish*, **Mish** is a Self Regularized Non-Monotonic Neural Activation Function, and its mathematically formula is presented as 

\begin{align*}
f(x)=x \tanh \left(\ln \left(1+e^{x}\right)\right)
\end{align*}

and it can be also presented by using *SoftPlus* function as 

\begin{array}{l}
f(x)=x \tanh (s(x)) \newline
s(x)= SoftPlus(x)=\ln \left(1+e^{x}\right)
\end{array}

Its $1_{st}$ and $2_{nd}$ derivatives are

\begin{aligned}
f^{\prime}(x) &=\frac{e^{x} \omega}{\delta^{2}} \newline
f^{\prime \prime}(x) &=\frac{e^{x}\left(e^{x}(4 x+6)+4 e^{x}+8 e^{2 x}+3 e^{3 x}+4\right)}{\delta^{2}}-\frac{2 e^{x}\left(2 e^{x}+2 e^{2 x}\right) \omega}{\delta^{3}}+\frac{e^{x} \omega}{\delta^{2}}
\end{aligned}

where

\begin{aligned}
&\delta=2 e^{x}+e^{2 x}+2 \newline
&\omega=4(x+1)+4 e^{2 x}+e^{3 x}+e^{x}(4 x+6)
\end{aligned}


The Taylor Series Expansion of *Mish* at $0$ and $\inf$ is calculated respectively as 

\begin{align*}
& 3 x / 5+8 x^{2} / 25-2 x^{3} / 125-86 x^{4} / 1875-7 x^{5} / 18750+O\left(x^{6}\right)  \newline
& \frac{e^{x}\left(e^{x}\left(x+O\left((1 / x)^{1 / 7}\right)\right)+\left(2 x+O\left((1 / x)^{1 / 7}\right)\right)\right)}{2 e^{x}+e^{2 x}+2}
\end{align*}

Therefore, *Mish* ranges between $-0.31% to infinity. *Mish* is largely used in any kind of benchmark test, and it has been proved empirically as a better choice of Swish or ReLU by providing smoother gradients. The only drawback we can see already from the formula given above is that it is definitely more computationally expensive than either Swish or ReLU.


# Short summary

At present, there are a total of 50+ types of activation functions, but there is no unified or clear standard to measure the quality of activation functions. The most fundamental reason is that the neural network through training is essentially to fit the non-linear distribution of training data. However, in reality, the distribution of data cannot be described by statistics, so it is impossible to deduce which activation function can be fitted better by theory, so most papers are tested by dozens of empirical studies for judgment. In spite of that, it is still able to see that a good activation function should have three properties

- $\color{red}{Unboundedness}$ (x > 0): Both *Sigmoid* and *Tanh* have saturated area, an appropriate initialization is needed to guarantee the input data distribute in unsaturated region. Otherwise, the gradient generated in the saturated region is too small will affect the convergence rate. 
- $\color{red}{Negatives}$: ReLU has dead issue by setting all negatives to be zero. Its variants, e.g., *LeakyReLU*, *PReLU*, *ELU*, *GLU*, *Swish*, *Mish*, *GELU* and so on allow partial negative outputs to improve the robustness of the model
- $\color{red}{Smoothness}$: *Relu*, *PReLU*, and *RReLU* are all step functions. The most obvious characteristics of the step function is the fault which is not as good as smooth activation function e.g., *Swish*, *GELU*, and *Mish*, for gradient updates.


# Extension:

I found a masterpiece from a data scientist via github which has a great way of visualizing varieties of activation functions. Try to play with it. It might help you remember it more. Click [here](https://dashee87.github.io/deep%20learning/visualising-activation-functions-in-neural-networks/) to his website.


