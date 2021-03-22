---
title: "Loss, Error, and Double Descent Phenomenon"
date: 2020-08-04T10:10:03-04:00
draft: true
---

Given a well-split set of data, machine learning practitioners try their best to find the "optimal" model (with the right weights) based on the performance of the model on the validation set or/and test set e.g. accuracy, mean square error, etc. We do this rather than looking at performance on training data, mainly because of $\color{blue}{\text{"overfitting"}}$, also known as $\color{blue}{\text{"error chasing"}}$ . Statistically speaking, we assume that the relationship between the independent variable $Y$ and the dependent variable $X$ is formulated as

\begin{align*}
Y = f(X) + \epsilon
\end{align*}

, where $f(\cdot)$ is the assumed true model, and $\epsilon$ is the unachievable random error. We try our best to find the "optimal" model $\hat{f}(\cdot)$ to approximate $f(\cdot)$ instead of approximating Y directly, because if so, we are chasing the error $\epsilon$ as well. It intuitively suggests that a relatively simpler (less complexity) model leads to better generalization. 

**However, the trend in modern machine learning practice is to build a very large complexity model that fits the data (near-) perfectly**. Logically this is an obvious overfitting case, yet it achieves high accuracy on test data. $\color{red}{\text{Isn't that a contradiction?}}$ In the past, a well-tuned LSTM+CRF architecture (i.e., 2 million parameters) was mainstream solutions for sequence labeling, but now over-parameterized models such as BERT ($\ge 334$ million parameters) and RoberTa ($\ge 125$ million parameters) after few epochs fine-tuning can significantly improve performance by several percentage points. Does that seem like we are sort of "underfitting"? 

This kind contradiction is then addressed as $\color{blue}{\text{"double descent"}}$ phenomenon. Basically, the performance of a machine learning/deep learning model, i.e., transformers, CNN, etc., first improves then gets worse, and then improves again as the complexity of the model increases. In addition, Nakkiran et al. [2020] observed that double descent curves can also be presented as a function of number of epochs (more training).  