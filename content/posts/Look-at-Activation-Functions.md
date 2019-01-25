---
title: "Look at Activation Functions"
date: 2019-01-23
tags: ['deep learning', 'natural language processing']
categories: ['deep learning']
---

# Overview:


Activation functions play a crucial rule in neural networks because they are the nonlinearities which have been attributed to the success story of deep learning. At present, the most popular activation functions are **ReLU** and its extended work such as **LReLU**, **PReLu**, **ELU**, **SELU**, and **CReLU** etc. However, none of them is guaranteed to perform better then others in all applications, so it becomes fundamental to understand their advantages and disadvantages in order to achieve better performances in specific applications. This blog will first introduce common types of non-linear activation functions, and then I will introduce which to choose on challenging NLP tasks. 


# Properties

**In general**, activation functions have properties as followings:

1. **non-linearity**: The non-linear activations functions are used not only to stimulate like real brains but also to enhance the ability of representation to approximate the data distribution. In other words, it increases large capacity  of model to generalize the data better;
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

    <p>
    \begin{align*}
    \sigma'(x) &= - \frac{1}{(1 + e^{-x})^2} (-e^{-x}) \\
          &= \frac{1}{1 + e^{-x}} \frac{e^{-x}}{1 + e^{-x}} \\
          &= \frac{1}{1 + e^{-x}} \frac{1 + e^{-x} - 1}{1 + e^{-x}} \\
          &= \sigma(x)(1 - \sigma(x))
    \end{align*}
    </p>  

### Disadvantages:
* **Gradient Vanishing**: When $\sigma(x) \rightarrow 0$ or $\sigma(x) \rightarrow 1$, the $\frac{\partial \sigma}{\partial x} \rightarrow 0$. Another intuitive reason is that the $\max f'(x) = 0.25$ when $x=0.5$. That means every time the gradient signal flows through a sigmoid gate, its magnitude always diminishes by one quarter (or more);
* Non-zero centered output: Imagine if x is all positive and all negative, what result will $f'(x)$ has? It slowers the convergence rate;
* Slow: Exponential computation is relatively slower comparing to ReLu


## Tanh

To solve the non-zero centered output, **tanh** is introduced since its domain is from [-1, 1]. Mathematically, it is just transformed version of sigmoid:

$$ \tanh(x) = 2\sigma(2x -1) = \frac{1 - e^{-2x}}{1 + e^{-2x}} $$

### Advantages:
* Zero-centered output: Release the burden of initialization in some degree; Also, it fasters the convergence. 
* Easy derivatives: 

    <p>
    \begin{align*}
    \tanh'(x) &= \frac{\partial \tanh}{\partial x} = (\frac{\sin x}{\cos x})' \\
          &= \frac{\sin'x \cos x + \sin x \cos'x}{\cos^2 x} \\
          &= \frac{\cos^2 x - sin^2 x}{\cos^2 x}\\
          &= 1 - \frac{\sin^2 x}{\cos^2 x} = 1 - \tanh^2(x)
    \end{align*}
    </p>  

### Disadvantages:
* Gradient Vanishing: When $\tanh(x) \rightarrow 1$ or $\tanh(x) \rightarrow -1$, $\tanh'(x) \rightarrow 0$
* Slow: Exponential computation is still included

## ReLU

**ReLU** has become the most popular method in deep learning applications. The idea behind is very simple, 

$$ReLu(x) = \max(0, x)$$

#### Advantages:
* Solves gradient vanishing problem
* Faster computation leads to faster convergence
* Even simpler derivative


#### Disadvantages:
* Non-zero centered
* **Dead ReLU problem**: Some of the neurons wont be activated. Possible reasons: 1. Unlucky initialization 2. Learning rate is too high. (Small learning rate, Xavier Initialization and Batch Normalization help).


## LReLU and PReLU

To solve ReLU problems, there are few work proposed to solve dead area and non-zero centerd problems.

### LReLU
* $f(x) = max(bx, x)$
* Normally, b = 0.01 or 0.3

### PReLU
* $f(x) = max(\alpha x, x)$
* $\alpha$ is a learnable parameter

Note: Even both methods are designed to solve ReLU problems, it is **NOT** guaranteed they will perform better than ReLU. Also, due to the tiny changes, they do not converge as fast as ReLU.


## ELU

What slows down the learning is the bias shift which is present in ReLUs. Those who have mean activation larger than zero and learning causes bias shift for the following layers. **ELU** is designed as an alternative of ReLU to reduce the bias shift by pushing the mean activation toward zero. 

<p>
\begin{split}
    ELU(x) &= \alpha (\exp(x) - 1), && \quad x \le 0 \\
           &= x, &&  \quad  x > 0   
\end{split}
</p>


### Advantages:
* Zero-Centered outputs
* No Dead ReLU issues
* Seems to be a merged version of LReLU and PReLU

#### Disadvantages:
* Slow
* Saturates for the large negative values


## SELU

The last common non-linear activation function is **SELU**, scaled exponential linear unit. It has self-normalizing properties because the activations that are close to zero mean and unit variance, propagated through network layers, will converge towards zero mean and unit variance. This, in particular, makes the learning highly robust and allows to train networks that have many layers.

<p>
\begin{split}
    SELU(x) &= \lambda \alpha (\exp(x) - 1), && \quad x \le 0 \\
           &= \lambda x, &&  \quad  x > 0   
\end{split}
</p>

which has gradient 

<p>
\begin{split}
    \frac{\partial d}{\partial x}  SELU(x) &= SELU(x) + \lambda \alpha, && \quad x \le 0 \\
           &= \lambda, &&  \quad  x > 0   
\end{split}
</p>

where $\alpha = 1.6733$ and $\lambda = 1.0507$.

***Question***: Would SELU, ELU be more useful than Batch Normalization?


# Activation functions on NLP

Here, I will list a few activations used on state-of-the-art NLP models, such as BERTetc.

## GELU 

Since BERT was released in December, all the NLP tasks benchmark scores have been updated, such as SQuad machine understanding, CoLLN 2003 named entity recognition, etc. By exploring tricks and theory behind BERT, BERT uses **GELU**, Gaussian error linear unit. Essentially, GELU uses a random error follows Gaussian distribution.

```
def gelu(input_tensor):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    input_tensor: float Tensor to perform activation.
  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf
```

<br>


# Extension:

I found a masterpiece from a data scientist via github which has a great way of visualizing varieties of activation functions. Try to play with it. It might help you remember it more. Click [here](https://dashee87.github.io/deep%20learning/visualising-activation-functions-in-neural-networks/) to his website.


