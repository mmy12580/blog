---
title: "Variant of Transformers"
date: 2020-01-06T12:51:02-05:00
draft: true
---

Transformers has been recognized as a mainstream architecture for all kinds of NLP tasks, such as transformer XL, Bert, Open-GPT2, and so on. 

![](https://static-asset-delivery.hasbroapps.com/a9e79c9b34ea183cad07eb995c5f51818b6c9447/9377375644a25fe0d886aece737b89dc.png)

The main reasons why transformers are preferred are as followings,

1. Long dependency
2. Semantic feature extraction 
3. Complex feature representation
4. Parallelism issue due to back-propagation through time (BPTT)

In other words, transformers are proposed to take advantage of speed part, __CNN__, and long dependency part, __RNN__, and it shows the power of complex representation based on `deeper` layers. 

As every NLPer knows, transformers are used in any kind of downstream tasks in NLP/NLU/NLG, especially pre-training models. However, in real applications, we are always looking for a specific `project metric` to reach, __speed__, __accuracy__ or __speed__ & __accuracy__. Given such, classic transformer won't be helpful for all cases. Thus, I am writing this blog to list variety of transformers given my best knowledge. 


## Self-Attention with Relative Position Representations

* __Idea__: extends the self-attention mechanism to efficiently
  consider representations of the relative positions, or distances between sequence elements, __Relation-aware Self-Attention__.

  - Original self-attention:

  	\begin{align}
		z_{i} &= \sum_{j=1}^{n} \alpha_{i j}\left(x_{j} W^{V}+a_{i j}^{V}\right) \newline
		\alpha_{ij} &= \frac{\exp e_{ij}}{\sum_{k=1}^{n} \exp e_{ik}} \newline
		e_{i j} &= \frac{\left(x_{i} W^{Q}\right)\left(x_{j} W^{K}\right)^{T}}{\sqrt{d_{z}}}
	\end{align}		
  
  - Relation-aware Self-Attention: change (1) to (4) and (3) to (5). Simplely put, adding edge information for more complex representation. Illustraion is like below, figure 1.

	![](/post_imgs/relation_aware_att.png)

	\begin{align}
		z_{i} &= \sum_{j=1}^{n} \alpha_{i j}\left(x_{j} W^{V}+a_{i j}^{V}\right) \newline
		e_{i j} &= \frac{x_{i} W^{Q}\left(x_{j} W^{K}+a_{i j}^{K}\right)^{T}}{\sqrt{d_{z}}}
	\end{align}

  - Relative Position Representations：Author hypothesized that __precise relative position information is not useful beyond acertain distance__. Clipping the maximum distance also enables the model to generalize to sequence lengths not seen during training. Thus, 2k + 1 unique edge labels are considered, then relative position representations, $w_K = (w^K_{-k}, \cdot, w^K_k)$ and $w_V = (w^V_{-k}, \cdot, w^V_k)$ are learned. $\color{blue}{test((((((()))))))}$

  \begin{align*}
	  a_{i j}^{K} &= w_{\operatorname{clip}(j-i, k)}^{K} \newline	
	  a_{i j}^{V} &= w_{\operatorname{clip}(j-i, k)}^{V} \newline
	  \operatorname{clip}(x, k) &= \max (-k, \min (k, x))
  \end{align*}



* __Advantage__: 
	- alllows transformer to adapt to unseen sequence length. A language model (LM) trained on seq_len 128 can be used to make inference on a sequence of size 256 without lossing performance
	- to save computation, every head shares same encoding among same layer
* __Findings__: 
	- combining relative and absolute position representations yields no further improvement in translation quality.
* __Extensions__: The paper shows that the relative postion can be treated as relation between words, for Graph Neural Network, relative postion is message function for modeling based on relation. Other relations are considerable for transformers, i.e., coref, dependency, pos-tagging, and relation in knowledge graph (kg). 
* __paper__: https://arxiv.org/pdf/1803.02155.pdf


## Transformer-XL

* __motivation__: Tranformers are limited by a fixed-length context in the setting of language modeling. The consequence of fixed-length is that the model cannot capture any longer-term dependency beyond the fixed context length. In addition, the fixed-length segments are created by selecting a consecutive chunk of symbols without respecting the sentence or any other semantic boundary. Hence, the model lacks necessary contextual information needed to well predict the first few symbols, leading to inefficient optimization and inferior performance. This problem is refered as `context fragmentation`.

* __advantage__:

	- Better performance due to a new positional encoding scheme

	- formula 


 
* __paper__: https://arxiv.org/pdf/1901.02860.pdf



