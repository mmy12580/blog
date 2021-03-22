---
title: "Transformer Component Analysis"
date: 2021-03-02T11:40:18-05:00
tags: ['natural language processing', 'deep learning']
categories: ['nlp', 'deep learning']
---
# Introduction
Although Transformers have become the state-of-the-art method in neutral language modeling, it is still unclear how each intermediate component contributes to the model performance. The pre-training and fine-tuning approach has been widely accepted, however the performance can differ greatly among datasets, along with the possibility of exhibiting poorer performance than some small capacity models like CNN or Bi-LSTM. Recently, many efforts have been made to transformers, mostly in the following three areas:

- Within Transformers
  - Adaptation to long sequences
  - Multi-head attention explains
  - Better representation by using transformations such as convolution, gated lining units, multi-branching units, DeLighT, etc.
- Model scaling
  - Width and depth control 
- Efficiency Improvement
  - Better token-level representations, i.e. BPE, adaptive inputs and outputs, DeFINE outputs.
  - Compression
  - Distillation
  - Pruning

The above work exposes some shortcomings in the particular transformer component as a result of the analysis and improvements made on that component, but there is no explanation with regard to how each component contributes to the model outputs and its reasonable performance. An [article](https://arxiv.org/abs/2011.03803) came to my attention recently that explains the components of transformers very well. In the next section, I will introduce to you the basic gist of the article.

# Component Analysis 

Transformers generally consist of a few stacked components, like `encoder-attention`, `encoder-decoder attention`,  `decoder-attention`, and `feedforward layers`. To address the importance of each component, two intuitive methods are proposed:  $\color{blue}{\text{contribution in information flow}}$ and $\color{blue}{\text{criticality in representation generalization}}$.

## Contribution in Information Flow

Transformer was originally used to study neural machine translation, in which information flow refers to how source text is translated into target text. Intuitively, an ablation experiment can be used to find out which part helps or hinders the information flow, and what we need to find is a metric that can be used to evaluate its importance. It is very similar to the early days of statistics where the Bayesian Information Criterion (BIC) was used to select the best subset of covariates for stepwise regression. More specifically, a backward feature selection procedure estimates the amount of change in BIC when the coefficient of a random feature is set to zero. Equivalently, researchers replaced the output of each component of a trained Transformer with zero and analyzed the performance of a resulting masked Transformer. A component is important if the performance without it is significantly worse than that of the full model; otherwise, it is redundant with the rest of the model.
Researchers address the contribution score of $n$-th component as a function of


\begin{align*}
    Contri_n = \frac{\widehat{M}_{n}}{\widetilde{M}} \quad where \quad \\
    \widehat{M}_n=
      \begin{cases}
        0 & M_{n} < 0 \\\\
        M_{n} & 0< M_{n} < C \\\\
        C & M_{n}>C
      \end{cases}, \quad
  \widetilde{M} = \max \\{ \widehat{M}_1, \cdots, \widehat{M}_N \\}
\end{align*}

where $M_n$ is the BLEU drop by ablating the $n$-th component, and the constant value C is (refer to the paper) recommended to be 10% of the baseline model BELU fraction. Simply put, the contribution score of $n$-th component is calculated by taking the maximum BELU change among all components in addition to its own BELU change. In the paper, the author refers to this score as a **hard** metric since it sets the output of a component towards zero.

## Criticality in representation generalization

Much like the contribution score in information flow, the criticality in representation generalization assesses the importance of components in terms of their impact on model performance. However, this approach measures the amount of re-useable components while keeping model performance, it is considered a soft metric by the author. *Module criticality phenomenon* is originally showed in [1] & [2]. They define the criticality of the module in a quantitative manner that depends on how close it is to the initial weights while maintaining performance for a convex combination of initial and final weights. Mathematically, its setting is as followings:

$$
  \theta_{n}^{\alpha_{n}}=\left(1-\alpha_{n}\right) \theta_{n}^{0}+\alpha_{n} \theta_{n}^{f}, \quad \text { where } 
  \alpha_n \in [0, 1] \\\\
  Criti_n =\min \alpha_{n} \quad s.t. \quad BLEU(f;\theta_n^f) - BLEU(f;\theta_n^{\alpha_n}) < \epsilon, f = Model
$$

where $\theta_n$ is the convex combination between initial weights $\theta_n^0$ and the final weights $\theta_n^f$, and  $BLEU(f; \theta)$ is the BELU score for the model $f$ given the parameters $\theta$. It is clear that the critical score for $n$-th component: $Crti_n$ is just the minimum $alpha$ to maintain the performance drop within a threshold value $\epsilon$. The small critical score of the $n$-th component means that we can move the weight of the $n$-th component far away for initialization without hurting the model performance. In the paper, $\epsilon$ is suggested as 0.5 BLEU point. Figure below shows an example.

![](/post_imgs/criticality.png)


## Component Importance Identification

Now, it is the fun part. To have a consistent format, I am going to use the symbols in the paper to represent those components. Let us do it in a dictionary way:

```
{
  "E:SA" : "Encoder Self-Attention",
  "E:FF" : "Encoder FeedForward",
  "D:SA" : "Decoder Self-Attention",
  "D:EA" : "Decoder-Encoder Attention",
  "D:FF" : "Decoder FeedForward"
}
```

A series of experiments have been done, and some results are shown in the figure below. Two metrics agree well with each other, and reveal several observations in common:

1. In general, the decoder-attention ("D:SA") layers are least important, and the $\color{red}{\text{decoder feedforward ("D:FF") layers are most important}}$.
2. Lower components in encoder (e.g. “E:SA” and “E:FF”) and higher components in decoder (e.g. “D:EA” and “D:FF”) are more important.
3.  $\color{red}{\text{Higher encoder-attention (“D:EA”) layers in decoder}}$ are more important than lower encoder attention layers. This is the same in [4] which claims that lower part of decoder is more like a language model. For the other components, the bottom and top layers are more important than the intermediate layer.
4. $\color{red}{\text{1-3 remain invariant with different initialization seeds}}$
5. $\color{red}{\text{1-3 hold in various model capacities}}$.

![](/post_imgs/transformer_components.png)

The authors also attempt *LayerDrop*[3], a form of structured dropout, to layer-wise ablate the experiments in addition to ablating each component. LayerDrop discards the entire component during training, which makes the network more robust to subsequent pruning. The results confirm their contention that the performance of different components of different layers varies. 

The  $\color{blue}{\text{most interesting}}$ thing to me was that the authors went on to explain why some components were considered unimportant. Several approaches were used to find the reasons for the presence of insignificant components, including $\color{blue}{\text{representation similarity analysis}}$, $\color{blue}{\text{learning dynamics analysis}}$, and $\color{blue}{\text{layer-wise isometry checks}}$.

![](/post_imgs/unimportant_components.jpeg)

It is able to draw conclusions from above figure that $\color{red}{\text{lower dropout rates and more training data resulted in fewer unimportant components}}$. In general, the lower the dropout rate, the lower the number of unimportant components in the model. It is plausible that a higher dropout rate results in a trained model with more redundant components, which are then more easily pruned without degrading performance. As the dataset grows larger, more components are needed. This is in line with the pre-training setup: training a 12- or 24-layer model with a large-scale dataset.

In addition, the authors present three simultaneous post-observation results.
- The output of unimportant components and the representation of the output layer are not similar
- [5] & [6] show that unimportant components can be identified in the early stage of training.
- Unimportant components are not due to deficient training.

## Group Component Important Identification

After describing different ablation experiments, the authors wondered what would happen if several components were ablated simultaneously. They did iteratively ablate multiple components from a trained transformer model, and report the BELU score of ablated model as below figure. In the early three ablation modules, the performance is unaffected by ablation, but as more areas are ablated, the performance declines rapidly. This can be easily understood as the interaction term in a simple linear regression. Given a two variable linear regression model $f(x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2 + error$ where $x_1 x_2$ is the interaction feature, it is possible that the hypothesis testing or generalization of test data may indicate that individual features are not as significant, however the interaction terms perform well.

![](/post_imgs/ablation_components.png)

**The question then arises**, if we identify unimportant components, but cannot remove them directly, is there a way to reasonably ”eliminate“ them and achieve better generalization? Thanks to the authors, and they utilize unimportant components to improve model performance with two strategies, namely $\color{blue}{\text{component pruning}}$ and $\color{blue}{\text{component rewinding}}$.

## References
[1] Chiyuan Zhang, Samy Bengio, and Yoram Singer. 2019. Are all layers created equal? ICML Workshop. \
[2] Niladri Chatterji, Behnam Neyshabur, and Hanie Sedghi. 2020. The intriguing role of module criticality in the generalization of deep networks. In ICLR. \
[3] Angela Fan, Edouard Grave, and Armand Joulin. 2020. Reducing Transformer Depth on Demand with StructuredDropout. ICLR. \
[4] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. 2019. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In ACL. \
[5] Namhoon Lee, Thalaiyasingam Ajanthan, Stephen Gould, and Philip H. S. Torr. 2020. A signal propagation perspective for pruning neural networks at initialization. ArXiv, abs/1906.06307. \
[6] Haoran You, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Yingyan Lin, Zhangyang Wang, and Richard Baraniuk. 2020. Drawing early-bird tickets: Towards more efﬁcient training of deep networks. ICLR.
