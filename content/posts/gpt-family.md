---
title: "Talk about the GPT Family"
date: 2020-06-16
tags: ['natural language processing', 'deep learning']
categories: ['nlp', 'deep learning']
---

Ever since pre-trained language model (PLM) has become the mainstream method for NLP applications, it is arguable whether to use Auto-Regressive (AR) or De-noising Auto-Encoder (DAE) models in order to achieving better performance in downstream tasks, such as classification, question and answer (Q & A) machine, multiple choice questions, auto summarization and so on. Certainly, other work has been proposed that combines the advantages of both types of models, i.e., XLNet, MASS, UniLM, etc. What is evident for sure from recent empirical work is that there is a growing effort to adapt the ___`bi-directional`___ transformer (AE) rather than the ___`uni-directional`___ transformer (AR) as ___`backbone`___, and one of the main reason is that bi-directional transformers can extract content information, whereas uni-directional transformers can only extract information from left to right, as below figure presents. 

![](https://www.researchgate.net/publication/334413801/figure/fig1/AS:779821865000960@1562935429820/The-Difference-Between-BERT-and-Open-GPT-extracted-from-Devlin-et-al-2-Figure-1.ppm)


Does that mean there is no future of uni-directional PLM models, such as GPT or GPT-2? $\color{red}{\text{Absolutely not!!!!!}}$ The newest paper released on June 3, 2020, [GPT-3](https://arxiv.org/abs/2005.14165) still adapts the uni-directional transformer, and it achieves state-of-the-art (SOTA) results. More importantly, GPT family can perform down-stream tasks in $\color{red}{\text{zero-shot}}$, $\color{red}{\text{one-shot}}$, or few shot settings – without any parameter or architecture modification. In this blog, I will mainly talk about GPT (generative pre-training) and its evolution, and I will share some thoughts as summary at the end. 


## Framework

General transformer structure contains a stack of encoders and decoders, and encoder-decoder attention is applied to connect them. In GPT, only encoders (paper called `decoders`) are used as figure showed below. Unlike normal encoders, GPT takes $\color{blue}{\text{masked multi-head attention}}$ mechanisms instead of multi-head attention. 

![](/post_imgs/gpt.png)


As a matter of fact, $\color{red}{\text{including language modeling as an auxiliary objective}}$ in fine-tuning stage helps (1) improving generalization of the supervised model, and (2) accelerating convergence. In the ablation study, it suggests that larger datasets benefit from the auxiliary objective but smaller datasets do not. Specifically, the overall objective is optimized as 

\begin{align*}
\sum_{(x, y)} \log P(y | x^{1}, \ldots, x^{m}) + \lambda \sum_{i} \log P(x_{i} | x_{i-k}, \ldots, x_{i-1}),
\end{align*}

where the left part of the plus sign is the objective of the target task, and the right part of the plus sign is the objective of the language modeling. After pre-training stage, GPT takes supervised fine-tuning for target tasks. Besides, $\color{blue}{\text{inputs need to be transformed for specific tasks}}$. The details is shown as following 

![](/post_imgs/gpt_input.png)


In 2019 February, an improved version of GPT so called GPT-2 is released. The idea behind the [paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) is to find out if it is possible to be as beneficial as multi-task learning fashion, but without explicitly implementing multi-task learning. It turns out that language models are unsupervised multi-task learners. Hence, it is valuable to apply language models for pre-training. 

Here, let us talk about **changes** from GPT to GPT-2:

1. GPT-2 alters its transformer by adding layer normalization before each sub-block and after self-attention. 
	- ![](/post_imgs/gpt2.png)
2. The second stage of supervised fine-tuning is replaced with unsupervised training of specific NLP tasks, thus making the structure of pre-training and Fine-tuning identical;
	- $\color{blue}{\text{How to do other language tasks rather than language modeling?}}$ Adding task-specific token after text. For example, paragraph + "TL：DR" is used for summarization. 
	- $\color{blue}{\text{Why not supervised fine-tuning?}}$ With this modification, the knowledge learned in GPT-2 is highly generalizable. (before GPT-3 out answer)
3. More data: 40GB crawled data;
4. Deeper and wider network.


## Deeper thought: GPT-3

In above section, I wrote a question $\color{blue}{\text{Why not supervised fine-tuning?}}$ in GPT-2. I have sort of seen what the authors wanted to prove, which was that the knowledge learned in the first phase of pre-training on a large unsupervised corpus could be generalized on an unsupervised task. About a month ago, they came out with the latest version of GPT, GPT-3, a 175 billion parameter autoregressive Language model. In the abstract of the paper, they have given the answer.

> ***While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do.***

In other words, GPT-3 focus on developing a more $\color{red}{\text{generic}}$ NLP model by requiring $\color{red}{\text{less domain data}}$ and $\color{red}{\text{no gradient updates or fine-tuning}}$ for downstream tasks. This is a **very very very** interesting finding; however to achieve that, it costs huge data (700GB), space and money (approximately 12 million ！！！). Only few companies in the world are feasible to do that.


To distinguish from terms of few-shots, authors specifically define as below. The illustration plot is also given for usage

1. **Few-Shot**: Few examples (normally 10-100) are given for demonstration but does not allow for gradients updates. The main advantage of it is that it greatly reduces the need for task-specific data and reduced likelihood of over-fitting. The main drawback is that so far the results of this approach are much worse than the latest fine-tuning models. Also, a small amount of task-specific data is still required.
2. **One-Shot**: Only one example is allowed for demonstration. Gradient updates are not allowed.
3. **Zero-Shot**: No illustrative examples. 


![](/post_imgs/gpt3_few_shots.png)

Let us check the performance. 

![](/post_imgs/gpt3_per.png)

The first plot shows the few-shot setting of a simple task, which requires the model to remove extraneous symbols from a word. Model performance improves with the addition of natural language task descriptions and $K$ number of examples in the model context. In addition, few-shot learning improves dramatically with model size. 

![](/post_imgs/gpt3_avg.png)

The second plot illustrates the average performance over different settings with model sizes. Zero-shot performance steadily increased as model size increased, while few-shots performances improve rapidly. They are more experiments have implemented, here is only few selected for illustration. Go check the [paper](https://arxiv.org/abs/2005.14165) for more, or go to their waiting list for trying their [openai-api](https://openai.com/blog/openai-api).


## Summary

All in all, GPT-3 was a very interesting attempt in NLP literature. From GPT to GPT-3, it is evident that the authors have conducted an in-depth study of the language model and the uni-directional transformer framework. In the mean time, the authors also incorporate language models with some mainstream ideas, such as multi-task learning and universal representational learning. In a way, you can see it as an attempt to hit the limits of the language model. The GPT-3 offers "new" ways (e.g., zero-shot, one-shot, and few-shot settings) of using PLMs compared to other pre-training models, and its methods are more applicable for comparison with humans. What did we learn from those work?

1. It takes an extremely large amount of data to get a generic NLP model. The larger the model, the better it performs in a few-shots settings. It is going to be very interesting to investigate the performance on bi-directional models with the same scale of GPT-3.
2. Auto-regressive models are further proved to be very helpful in some NLU/NLG tasks. Check their benchmark on experiments.
3. Limitations
	- **Notable weakness in text synthesis** and several NLP asks. In terms of text synthesis, while the overall quality is high, the GPT-3 sample still sometimes repeats semantics at the file level, in sufficient Long paragraphs begin to lose coherence and contradict themselves. For some other NLP tasks, GPT-3 has difficulty to handle it. 
	- **Structure objective**: Autoregressive models are lagging few-shot performance on a few of the tasks due to its structure. Unlike bi-directional models, they naturally adapt tasks such as fill-in-the-blank, probing, and some difficult Q & A machine. 
	- **Fundamental limitation**: Scaling up LM-like models will eventually run into the limits of pre-training objective. In general, with self-supervised objectives, the task specification relies on forcing the required task into a prediction problem. In fact, a useful language system may be better thought of as taking $\color{blue}{\text{goal-directed action}}$ rather than merely making predictions[4]. In NLG tasks, there are a lot of work already showing that maximizing the likelihood degenerates. Check my previous [post](https://moyan.ml/posts/decoding_2020/) for NLG decoding. 
	- **Poor sample efficiency**: Even though, GPT-3 mimics the behavior of human in zero-shot or one-shot settings, the text it sees during pre-training are much more than human sees in their lifetime[5].
	- **Real few-shot learning?**: It is not clear whether few-shot learning actually learns new tasks from scratch at inference time, or if it simply recognizes and identifies tasks that it has learned during training. 
	- **Even harder to interpret**: Given the scale of GPT3, it gets us further and further away from getting an interpretable model. Although some work has shown that transformers are a variant of graphical neural networks (GCNs) that can provide some interpretability and causality, more efforts are expected.


At the end, while pre-trained models provide a good paradigm (e.g., fine-tuning or few-shot learning) for NLP applications, we are still looking for a great solution which is simpler, interpretable, robust, and generalizable.


## Reference

[1] Radford, Alec and Narasimhan, Karthik and Salimans, Tim and Sutskever, Ilya Improving Language Understanding by Generative Pre-Training. OpenAI, 2018.

[2] Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya. Language Models are Unsupervised Multitask Learners. OpenAI, 2019.

[3] Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei. Language Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165, 2020

[4] Adam Roberts, Colin Raffel, and Noam Shazeer. How much knowledge can you pack into the parameters
of a language model? arXiv preprint arXiv:2002.08910, 2020.

[5] Tal Linzen. How can we accelerate progress towards human-like linguistic generalization? arXiv preprint arXiv:2005.00955, 2020.