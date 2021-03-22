---
title: "NLG Decoding Strategies"
date: 2020-05-07
tags: ['deep learning', 'natural language processing']
categories: ['natural language processing', 'deep learning']
---
# Generic Issue

Although the development of pre-trained methods have led to a qualitative advance in the field of natural language modeling, the quality of natural language generation continues to be questionable. One of the main reasons found in empirical study (Holtzman et al., 2019) is that maximization-based decoding methods leads to **degeneration**. In other words, the output text is **bland**, **incoherent**, or in a **repetitive** cycle. These problems can't be solved by simply increasing the amount of training data, e.g., large-scale GPT-2 (Radford et al., 2019) present the same issues. In the next part of this blog, I will $\color{red}{\text{focus on}}$ how decoding strategy helps. In the extension section, I will mention other methods than decoding strategies for better language generation. 


# Deterministic Decoding

Two widely used deterministic decoding approaches are $\color{blue}{\text{greedy search}}$ and $\color{blue}{\text{beam search}}$. The most intuitive decoding approach is greedy search due to the assumption of language model that the probability distribution of a word sequence can be decomposed into the product of conditional next word distributions:

\begin{align*}
P(w_{1}:_{T} | W_{0})=\prod_{t=1}^{T} P(w_{t} | w_{1: t-1}, W_{0})
\end{align*}

where $W_0$ is the initial context word sequence. Greedy search simply selects the word with highest probability as its next word at each timestep $t$.

\begin{align*}
w_t = argmax_wP(w|w_{1:t-1})
\end{align*}

One obvious drawback of greedy search is that it often produces a sub-optimal output sequence by missing high conditional probability words, e.g., $p(w_{32} | w_{22}) > p(w_{31} | w_{21})$ with $p(w_{22}) < p(w_{21})$. Here, $w_{ij}$ is the candidate word $j$ at timestep $i$. 

**Beam search** is then introduced to reduce the risk of omitting hidden high probability word sequences by retaining the most likely hypothesis at each time step and ultimately selecting the hypothesis with the highest overall probability. Mathematically, at time step $t$, beam search maintains a set of $K$ hypothesis $\mathcal{H}_t$:

\begin{align*}
\mathcal{H}_{t}= \\{ (w_{1}^{1}, \ldots, w_{t}^{1}), \ldots, (w_{1}^{K}, \ldots, w_{t}^{K}) \\}.
\end{align*}

Each hypothesis $\tilde(h)_{w_t^i}^i \in \\{1, \ldots, K \\}$ from $\mathcal{H}_t$ is expanded with all possible next tokens $v$ from the vocabulary $V$ to from candidate hypothesis, and it is formulated as

\begin{align*}
\tilde{h}_{v}^{i}=h_{w_{t}}^{i} \|(v)= (w_{1}^{i}, \ldots, w_{t}^{i}, v).
\end{align*}

The score assigned to it will be 

\begin{align*}
s_{\tilde{h}_{v}^{i}} = s_{\tilde{h}_{w_t}^{i}} + \log p(v | w_{1:t}^i)
\end{align*}

and a new hypothesis set of $K$ hypothesis is then constructed as 

\begin{align*}
\mathcal{H}_{t+1}=\underset{i, v}{\text{arg-top-k}} \space s_{\tilde{h}_{v}^{i}}
\end{align*}

At finaly, we generate a set of candidate output sequence $\mathcal{M}_t$:

\begin{align*}
\mathcal{M}_{t}= \\{h_{v}^{i} \in \mathcal{H}_{t+1} | v= \<eos\> \\}
\end{align*}

where $v= \<eos\>$ is the termination signal of beam search. Certainly, we can introduce other **early stopping** rules for beam search termination.


It is not hard to find that **beam search will always output a sequence with higher joint probability than greedy search, but it does not guarantee to output the most reasonable (fluent, no repetition) one**. 


## N-gram Penalty

A simple remedy is to add $\color{blue}{\text{n-gram penalty}}$ or $\color{blue}{\text{n-gram blocking}}$ to beam search, as introduced by Paulus et al. (2017) and Klein et al. (2017). The idea of it is that it makes sure that no n-gram appears twice by manually setting the probably of next words that could create an already seen n-gram towards 0. In coding, an extremely small value $-10e20$ will be applied for log probability instead.

```python 
def block_ngram_repeats(sequences, log_probs, block_ngram_repeat=2):
    total_path, cur_len = sequence.shape
    for path_idx in range(total_path):
        # skip BOS
        hyp = alive_seq[path_idx] # [batch_size, seq_len]
        ngrams = set()
        fail = False
        gram = []
        for i in range(cur_len - 1):
            # Last n tokens, n = block_ngram_repeat
            gram = (gram + [hyp[i]])[-block_ngram_repeat:]
            # print("gram:", gram) 
            # skip the blocking if any token in gram is excluded
            if set(gram) & exclusion_tokens:
                continue
            if tuple(gram) in ngrams:
                fail = True
            ngrams.add(tuple(gram))
        if fail:
            log_probs[path_idx] = -10e20
```

Although n-gram penalties can be used to avoid repeating n-grams, it can affect generation with specific n-grams, like `entities`, Yangzi River, British Columbia, and Macbook Pro. Finding a good trade-off between forced 'no-repetition' like n-gram penalty and looping the same n-grams over and over requires a lot of fine-tuning. Thus, we need better methods for decoding. 


## Variants of Beam Search

Holtzman et al. (2018) found that original beam search leads to __limited diversity__ in the beam and therefore cannot exploit the strength of models. Instead, they score the current hypotheses in the beam with the full decoding objective

- First, each hypothesis is expanded by selecting the $k$ highest scoring next words e.g., k = 10
- Second, $k$ sequences are sampled from the $k^2$ candidates according to the (Softmax normalized) distribution over the candidate scores given by the full decoding objective.
- At last, temperature is introduce.

The algorithm is illustrated as 

![](/post_imgs/holtzman_beam_search.png)


$\color{blue}{\text{Diverse beam search}}$(DBS) (Vijayakuma et al., 2018) produces sequences with significant variability by incorporating diversity constraints in the candidate sequence groups at decoding. In addition, it does this with minimal computational or memory overhead. 


![](/post_imgs/diverse_beam_search.png)

Number of groups $G$ and diversity strength $\lambda$ are hyper-parameters. When $G=1$, DBS becomes normal beam search. The diversity strength $\lambda$ specifies the trade-off between the model score and diversity terms. A higher value of $\lambda$ produces a more diverse list, however, very large $\lambda$ values can make the model score too high, resulting in grammatically incorrect output.


Kulikov et al. (2018) used $\color{blue}{\text{iterative beam search}}$, which relaxes the inclusion criterion. It ensures that partial hypothesis set of beam search in $l$-th iteration has minimum overlap with any part of the search space previously explored in the $l-1$ iteration of the beam search. After running multiple iterations, the best one is selected according to the log probability assigned by the model. 



# Stochastic Decoding

Alternative approach to deterministic decoding algorithm is sampling from the model at generation time. This means language generation is not deterministic anymore. Currently available sampling methods:

- Temperature Sampling 
- $\color{blue}{\text{Top-k sampling}}$ (Fan et al., 2018) 
- $\color{blue}{\text{Nucleus sampling}}$ (Holtzman et al., 2019)
- $\color{blue}{\text{Stochastic beam search}}$ (Kool et al., 2019)
- $\color{blue}{\text{Entamax sampling}}$ (Martins et al., 2020)


## Temperature Sampling

A temperature $t$ is included in Softmax to change the vocabulary probability distribution in favor of high probability words:

\begin{align*}
P(w | w_{1: t-1})=\frac{\exp (u_{t} / t)}{\sum_{t^{\prime}} \exp (u_{t^{\prime}} / t)}, \text { and } t \in [0,1)
\end{align*}

When $t$ -> $0$, it becomes greedy search; when $t$ -> $\inf$, it becomes uniform sampling. It can avoid sampling from tail distribution by finding an appropriate value of $t$.

## Top-K Sampling

A even more effective and simpler method so called **Top-k Sampling** is proposed. In generation stage, it selects top $k$ tokens with the highest probability first, and then transform the distribution by dividing its sum

\begin{align*}
P'(w | w_{1: t-1}) &= P(w | w_{1: t-1}) / p, \newline
p' &= \sum P(w| w_{1: t-1}) 
\end{align*}
At end, we draw samples from $P'(w | w_{1: t-1})$ as output tokens. The problem with Top-k Sampling is that $k$ is pre-given as a constant, and for sentences of varying length and size with different contexts, we may sometimes need more (or less) tokens than $k$. An quick example is given as below. The first prefix can be followed by diverse options, and at this point 10 tokens may not be enough to cover all possible options; while the second prefix only has few options followed, so that 10 tokens might be too much to lead the model draw samples from tail distribution.


```python
> k = 10
> prefix1 = "She said, 'I never"
> prefix2 = "I ate the pizza while it was still"

> top_k_candidates(prefix1, k)
[['thought', 'knew', 'had', 'saw', 'did', 'said', 'wanted', 'told', 'liked', 'got'],
[0.92, 0.934, 0.873, 0.834, 0.803, 0.720, 0.643, 0.539, 0.485, 0.433]]
> top_k_candidates(prefix2, k)
['hot', 'warm', 'cooling', 'on', 'heating', 'fresh', 'cold', 'warming', 'burning', 'cooking']
[0.903, 0.845, 0.833, 0.712, 0.644, 0.634, 0.587, 0.512, 0.435, 0.289]]
```

## Nucleus Sampling 

Nucleus Sampling, or Top-p Sampling (often called), is then proposed to solve the problem with Top-K Sampling as described above. In the Top-p sample, instead of sampling only the most likely K words, we select the smallest set of words with cumulative probability exceeding probability $p$ and redistribute the probability mass across this set of words. In this way, the size of the phrase (that is, the number of words in the phrase) can dynamically increase and decrease according to the probability distribution of the next word. 

Mathematically, Top-p sampling replaces $p' = \sum P(w| w_{1: t-1})$ from Top-K sampling with a constant defined as $p' \in (0,1)$. Let us define $p$ = 0.85, then the candidates will be ['thought', 'knew', 'had'] for the first prefix, and only ['hot'] for the second prefix from above examples.

In applications, Top-p and Top-k sampling can be used together. The source code built in `transformers` is provided as 

```python
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
```

Essentially, both Top-K sampling and Top-p sampling are sampling tokens from truncated vocabulary distribution, but the main difference is sampling from different confidence interval.


## Stochastic Beam Search


Another way to improve Top-k sampling is to implicitly apply the Gumbel-Top-k trick for sampling without replacement, also known as stochastic beam search. The algorithm establishes a theoretical link between sampling and (deterministic) beam search and can serve as a principled intermediate alternative. Algorithm is attached, and source code are be found in 

![](/post_imgs/sbs1.png)


[Click here for source code](https://github.com/wouterkool/stochastic-beam-search). 

## Entamax sampling 

Martins et al. (2020) points out all of above methods create a mismatch between training and testing conditions. To avoid this mismatch, they applied Entmax (sparse transformation and entmax sampling) to train and sample from a natively sparse language model like GPT-2. Entmax transforms a vector of scores into a **sparse probability distribution**, preventing implausible words from receiving any probability mass. In addition, it comes with a well-defined loss function that allows it to automatically learn its sparsity from the data during the training process.

Given a set $\mathcal{S}$ of training sentences, the usual strategy for learning the language model parameters $\theta$ is to minimize the negative log-likelihood:

\begin{equation}
\mathcal{L}(\theta)=-\sum_{i=1}^{|\mathcal{S}|} \sum_{t=1}^{T_{i}} \log p_{\theta} (x_{t}^{i} | x_{1:t}^{i}) 
\end{equation}

The standard option to model $p_{\theta} (x_{t}^{i} | x_{1:t}^{i})$ in equation (1) is to compute a score vector $z_t$ by conditioning on the context $x_{1:t}$, and then applying a Softmax transformation. In entmax setting, $\alpha$-entmax is used for sparse transformation, 

\begin{equation}
\alpha \text{-entmax} (z_t):= \underset{\boldsymbol{p} \in \Delta^d}{\operatorname{argmax}} p^T z_t + \mathrm{H}_{\alpha}(\boldsymbol{p}).
\end{equation}

where $\Delta^{d}:=\\{\boldsymbol{p} \in \mathbb{R}^{d} | \sum_{i=1}^{d} p_{i}=1, \boldsymbol{p} \geq \mathbf{0} \\}$ is the probability simplex, and $\mathrm{H}_{\alpha}$ is the Tsaills $\alpha$-entropy. Blondel et al. (2019) have shown that, for $\alpha > 1$, entmax is able to output sparse probability distributions, where some words get **exactly zero probability**, whereas softmax 
($\alpha = 1$) does not have this capability. By modifying Eq.(1) with $\alpha$-entmax, the negative log-likelihood loss then becomes

\begin{equation}
\mathcal{L}(\theta)=\sum^{|\mathcal{S}|} \sum^{T_{i}} \ell_{\alpha}(\boldsymbol{z}_{t}(\theta, x_{1:t}), x_{t})
\end{equation}

where $\ell_{\alpha}(\boldsymbol{z}_t, x)$ is the $\alpha$-entmax loss in Eq. (2). When $\alpha=1$, it recovers the negative log-likelihood; when $\alpha=2$, this corresponds to sparse-max loss. Entamax Sampling performs sampling from categorical distribution obtained by applying the entmax transformation to scores $z_t$ given by the model:

\begin{align*}
x_{t} \sim p_{\theta} (\cdot | x_{1:t}) = \alpha \text {-entmax }(z_t (\theta, x_{1:t}))
\end{align*}

Comparing to other sampling schemes, it does not require ad-hoc modification, and it considers a varying number of tokens depending on the context.

Available package in python, [click here for more](https://github.com/deep-spin/entmax).


# Conclusions and Extensions

In general, decoding objective based on maximization are not appropriate for open-ended text generation, e.g., storytelling, dialogue, etc. In other words, those models degenerate. As a comparison, stochastic decoding methods (Top-K, Top-P, SBS, Entmax) seem to produce more fluent text than deterministic search on open-ended language generation. One of the main reason deterministic decoding methods generate repetitive word sequence are caused by model, especially the way the model is trained, rather than decoding method (Welleck et al., 2019). Therefore, **beam search can generate more fluent text than Top-p sampling when adapting the model's training objective**. In later research of Welleck et al. (2020) found that stochastic decoding methods, Top-K and Top-p, also suffer from generating repetitive word sequence. **What can do we do then?**


Decoding strategies are helpful to rectify issues described in the introduction. However, the core issue is not addressed: **the model’s underlying sequence probabilities are clearly not correct**.
Welleck et al. (2019) show that the primary factor of text degeneration is the use of likelihood, and they proposed a new objective, **unlikelihood training**, which forces unlikely generations to be assigned lower probability by the model. For details, check [here](https://arxiv.org/pdf/1908.04319.pdf).



# References

[1] Romain Paulus, Caiming Xiong, and Richard Socher. 2017. A deep reinforced model for abstractive
summarization. arXiv preprint arXiv:1705.04304.

[2] Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart, and Alexander M Rush. 2017. Opennmt: Open-source toolkit for neural machine translation. arXiv preprint arXiv:1701.02810.

[3] Ari Holtzman, Jan Buys, Maxwell Forbes, Antoine Bosselut, David Golub, and Yejin Choi. 2018. Learning to write with cooperative discriminators. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1638–1649. Association for Computational Linguistics.

[4] Ashwin K Vijayakumar, Michael Cogswell, Ramprasaath R Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. 2018. Diverse beam search for improved description of complex scenes. In Thirty-Second AAAI Conference on Artificial Intelligence.

[5] Ilya Kulikov, Alexander H Miller, Kyunghyun Cho, and Jason Weston. 2018. Importance of a search strategy in neural dialogue modelling. arXiv preprint arXiv:1811.00907.

[6] Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. arXiv preprint arXiv:1805.04833

[7] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI Blog, 1(8).

[8] Ari Holtzman, Jan Buys, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751. International Conference on Machine Learning.

[9] Kool Wouter, Van Hoof Herke, and Welling Max. 2019.  Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement.

[10] Welleck S., Kulikov I., Roller S., Dinan E., Cho K., Weston J., 2019. Neural Text Generation with Unlikelihood Training. arXiv preprint arXiv:1908.04319.

[11] Martins P.H., Marinho Z., Martins A.F.T., 2020. Sparse Text Generation. arXiv preprint arXiv:2004.02644.

[12] Welleck S., Kulikov I., Kim J., Yuanzhe Pang R., Cho K., 2020. Consistency of a Recurrent Language Model With Respect to Incomplete Decoding. arXiv preprint arXiv:2002.02492.