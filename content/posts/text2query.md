---
title: "A quick summary: Text-to-SQL"
date: 2020-03-18T17:36:33-04:00
tags: ['deep learning', 'natural language processing']
categories: ['nlp', 'deep learning']
---

# Definition

Given relational database (or table), users provide the question, the algorithms generates the correct SQL syntax. Below example is an illustration from ___Spider Dataset___.

![](/post_imgs/text2query1.png)



# Public Dataset:

1. __`ATIS & GeoQuery`__: flight tickets reserving system
2. __`WikiSQL`__: 80654 training data
	- Single table single column query
	- Aggregation
	- Condition
	- Operation 
	- Github: https://github.com/salesforce/WikiSQL
3. __`Spider`__: a. more domains; b. more complex SQL syntax; 
	- Complex, Cross-domain and Zero-shot
	- Multi tables and columns
	- Aggregation
	- Join
	- Where
	- Ranking
	- SQL connection
	- Github: https://github.com/taoyds/spider
4. __`Sparc`__: Context-dependent and Multi-turn version of the Spider task
	- Home: https://yale-lily.github.io/sparc
5. __`CoSQL`__: Cross-domain Conversational, the Dialog version of the Spider and SParC tasks
	- Home: https://yale-lily.github.io/cosql	
6. Chinese related
	- Chinese __`Spider`__
		- Github: https://github.com/taolusi/chisp
	- __`TableQA`__
		- Chinese version WikiQA
		- Single table multiple columns
		- Aggregation
		- Condition
		- Operation
		- Github: https://github.com/ZhuiyiTechnology/nl2sql_baseline
	- __`DuSQL`__ (2020)
		- 200 Database and 23K (question, SQL query), 18K train, 2K validation, 3K test set
		- Home: https://aistudio.baidu.com/aistudio/competition/detail/30?isFromCcf=true		


__Statistics__:

![](/post_imgs/text2query2.png)


# Evaluation Metrics:

1. Exact match
2. Execution accuracy

Note that WikiSQL takes both, while Spider only uses first one.


# SOTA Models:

Basically, all models adapt an encoder-decoder mechanism. In encoder's direction, textual alignment or mapping is very important among NL question and database. In other words, the key information of  questions, e.g., entities, SQL condition, and others. On the other hand, decoder cleares the boundaries of semantics, and it makes SQL format consistent based on representations of natural language questions. Traditional seq2seq models are not able to learn based on above mentioned requirements. Thus, in general, streaming models are based on following aspects:

1. __More contextualized representation__ i.e., BERT, XLNET;
2. Better structure explicitly enhance alignment, such as GNN;
3. Improve accuracy of SQL execution by __scaling down search space__ e.g., tree-based decoding, and slots-filling decoding;
4. Provide __more abstract intermediate layers representation__;
5. Reranking returns based on defining __new feature alignment__;
6. Efficient data augmentation.


## 1. Pointer Network

Traditional seq2seq models employ decoders with fixed vocabulary. Unlike seq2seq models, text-to-sql outputs sequences may include a) words/phrases in question; b) SQL key words; c) Column and row names correspond to database.

Pointer Network solves above issues since its outputs vary given vocabulary based on inputs. More specifically, attention mechanism directly helps taking words from inputs to be part of outputs.

## 2. Seq2SQL

One of the main drawbacks from Pointer Network is that it does not use the information based on the query language. Seq2SQL divides the generated sequences into three parts, which are aggregation (e.g., sum, count, min, and max etc.), select, and condition where.

Select and aggregation parts employ attention mechanisms for text classification, and where condition adopts pointer network. However, where condition syntax is not unique, such as 

````sql
SELECT name FROM insurance WHERE age > 18 AND gender ="male";

SELECT name FROM insurance WHERE gender = "male"AND age > 18;
````

This can directly leads outputs to be incorrect given natural language question. Therefore, the author proposes a scoring system with RL over generated outputs.  

![](/post_imgs/text2query3.png)

## 3. SQLNet

Even though RL is a good idea to select the `best` results from candidate SQL syntax, it is not doing properly as expected. SQLNet introduces slots-filling tasks for select and where parts. Unlike Seq2SQL, SQLNet adapts `sequence-to-set` mechanism to select possible where condition candidates on target SQL syntax. It calculates the top k highest probability over candidates, and at final outputs values for symbols and conditions.


![](/post_imgs/text2query4.png)


## 4. TypeSQL

It considers an more obvious methods, block filling. The idea is to obtain the numbers and unregistered entities. Simply put, TypeSQL gives very word a pre-defined type. 

The process of recognizing:

1. Splits the question (Q) to n-grams (n from 2 to 6);
2. Search n-grams in database;
3. For matched cases, provides columns values, e.g., INTEGER、FLOAT、DATE、YEAR
4. For entities, search over `FREEBASE`, to identify 5 types: PERSON，PLACE，COUNTREY，ORGANIZATION，SPORT

TypeSQL consider dependency and similarity among key information, such as `SELECT_COL`, ` `COND_COL` and number of conditions (`#COND`). In summary, there are three independent models for predicting values in blocks:

1. `MODEL_COL：SELECT_COL，#COND，COND_COL`
2. `MODEL_AGG：AGG`
3. `MODEL_OPVAL：OP, COND_VAL`

![](/post_imgs/text2query5.png)


## 5. SyntaxSQLNet

All above methods outputs a linear text from decoder, while a tree based structure can be more informative. In the original paper, it is been proved to improve the accuracy about 14.8%.

![](/post_imgs/text2query6.png

SyntaxSQLNet takes 9 modules to predict the final SQL syntax, and each module represents one component of SQL syntax. Below are 9 modules:

1. __IUEN__: INTERCEPT、UNION、EXCEPT、NONE
2. __KeyWords__: WHERE、GROUP BY、ORDER BY、SELECT
3. __Column__: column names
4. __Operation__: >, <, BETWEEN, LIKE
5. Aggregation: MAX, MIN, SUM, COUNT, etc.
6. Root/Terminal/Module: sub-search and termination symbol
7. AND/OR: relationship between expressions
8. DESC/ASC/LIMIT: keywords related to ORDERBY
9. HAVING: Having condition expression

Besides, the authors also proposed a cross-domain augmentation method. The specific implementation wont be introduced here. Go check [here](https://github.com/taoyds/syntaxSQL/blob/master/generate_wikisql_augment.py) for `Python` implementation.


## 6. IRNet

Similar to SyntaxSQLNet, IRNet also defines a series of modules, and convert the SQL syntax as tree based structure. `SemQL` parses the SQL syntax as below

![](/post_imgs/text2query7.png)

Another improvement is called `scheme-linking`, which shows how to find cell values and columns from the given question. Author summaries the common entities as three categories, table names, column names, and cell values. Based on this three kinds entities, use a) n-gram to match table names and column names; b) Employed ConceptNet for related cell values and columns names. Illustration is as below

![](/post_imgs/text2query8.png)



## 7. Global-GNN

To better investigate structure information under relational database, BenBogin, et al. proposed a graph based neural network. As illustration below, bold circles represent table, while non-bold nodes represent column names. Bidirectional edges represent the subordinate relationship of tables and column names; red and blue edges represent primary and foreign key relationships. Orange nodes represent problem-related results, and light colors are irrelevant.


![](/post_imgs/text2query9.png)

In addition, they also propose a reranking algorithm based on global information, named as `global reasoning`. An example illustrated below is that it is not clear that the name points to the singer or the song, but we are able to observe nation shows up only in singer, so it should be singer. It is used to dissolve ambiguity. The pipeline of Global-GNN is

![](/post_imgs/text2query10.png)


## 8. RAT-SQL

Unlike Global-GNN, RAT-SQL adds questions nodes in the graph, and enrich edge types with string matching, e.g., regex. The new edges are given 

![](/post_imgs/text2query11.png)



## More? 

Absolutely yes. Below is what I found good to look at 

### Weakly Supervised

Do not apply SQL syntax as signals for training. 

`Paper`
- [ ] Min S, Chen D, Hajishirzi H, et al. [A discrete hard em approach for weakly supervised question answering](https://www.cs.princeton.edu/~danqic/papers/emnlp2019.pdf)[C]. EMNLP-IJCNLP 2019.  
- [ ] Agarwal R, Liang C, Schuurmans D, et al. [Learning to Generalize from Sparse and Underspecified Rewards](https://arxiv.org/pdf/1902.07198.pdf). 2019.  
- [ ] Liang C, Norouzi M, Berant J, et al. [Memory augmented policy optimization for program synthesis and semantic parsing](https://papers.nips.cc/paper/8204-memory-augmented-policy-optimization-for-program-synthesis-and-semantic-parsing.pdf)[C].NeurIPS, 2018: 9994-10006.


`Code` 
- [https://github.com/shmsw25/qa-hard-em](https://github.com/shmsw25/qa-hard-em)  
- [https://github.com/google-research/google-research/tree/master/meta_reward_learning](https://github.com/google-research/google-research/tree/master/meta_reward_learning)

`Score`  

|Hard-EM|84.4 |  83.9  |
|-|-|-|
|MeRL | 74.9 | 74.8 |
|MAPO | 72.2 | 72.1 |

### ExecutionGuided

Execution Guided filters out the non-sense sql syntaxes, and fix some minor errors.
Three common execution errors
1. Syntax Error
2. Run-time error, e.g., SUM over strings
3. Null errors, beam search could help 

`Paper`  
- [ ]  Wang C, Huang P S, Polozov A, et al. [Robust Text-to-SQL Generation with Execution-Guided Decoding](https://arxiv.org/pdf/1807.03100.pdf)[J]. 2018.  
- [ ]  Wang C, Brockschmidt M, Singh R. [Pointing out SQL queries from text](https://openreview.net/pdf?id=BkUDW_lCb)[J]. 2018. 
- [ ]  Dong L, Lapata M. [Coarse-to-fine decoding for neural semantic parsing](https://arxiv.org/pdf/1805.04793.pdf)[J]. 2018.
- [ ]  Huang P S, Wang C, Singh R, et al. [Natural language to structured query generation via meta-learning](https://arxiv.org/pdf/1803.02400.pdf)[J]. 2018.

`Code`  
- [https://github.com/microsoft/PointerSQL](https://github.com/microsoft/PointerSQL)  
- [https://github.com/donglixp/coarse2fine](https://github.com/donglixp/coarse2fine)  

`Score`  

|Coarse2Fine + EG| 84.0 | 83.8 |
|-|-|-|
| Coarse2Fine | 79.0 | 78.5 |
| Pointer-SQL + EG | 78.4 |  78.3  |
| Pointer-SQL | 72.5 | 71.9 |


### Extension to SQLNet 

`Paper`  
- [ ]  Xu X, Liu C, Song D. [SQLNet: Generating structured queries from natural language without reinforcement learning](https://arxiv.org/pdf/1711.04436.pdf)[J]. 2018.  
- [ ]  Hwang W, Yim J, Park S, et al. [A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization](https://arxiv.org/pdf/1902.01069.pdf)[J]. 2019.  
- [ ] He P, Mao Y, Chakrabarti K, et al. [X-SQL: reinforce schema representation with context](https://arxiv.org/pdf/1908.08113.pdf)[J]. 2019.

`Code`  
- [https://github.com/naver/sqlova](https://github.com/naver/sqlova)  
- [https://github.com/xiaojunxu/SQLNet](https://github.com/xiaojunxu/SQLNet)  

`Score`  

| BERT-XSQL-Attention + EG |92.3|91.8|
|-|-|-|
| BERT-XSQL-Attention | 89.5 | 88.7 |
| BERT-SQLova-LSTM|87.2 |  86.2  |
|BERT-SQLova-LSTM + EG | 90.2 | 89.6 |
|GloVe-SQLNet-BiLSTM|69.8 |  68.0  |


### Model Interactive
Iteratively generate correct sql by clarifying users intention.

`Paper`
- [ ] Yao Z, Su Y, Sun H, et al. [Model-based Interactive Semantic Parsing: A Unified Framework and A Text-to-SQL Case Study](https://arxiv.org/pdf/1910.05389.pdf)[C]. EMNLP-IJCNLP 2019.  

`Code`
- [https://github.com/sunlab-osu/MISP](https://github.com/sunlab-osu/MISP)

### EditSQL

`Paper`
- [ ] Zhang R, Yu T, Er H Y, et al. [Editing-Based SQL Query Generation for Cross-Domain Context-Dependent Questions](https://arxiv.org/pdf/1909.00786.pdf)[C]. EMNLP-IJCNLP 2019.

`Code`
- [https://github.com/ryanzhumich/editsql](https://github.com/ryanzhumich/editsql)

`Score`  

| EditSQL + BERT  |57.6|53.4|
|:-:|:-:|:-:|
| EditSQL | 36.4 | 32.9 |  


## References

[1] Seq2sql: Generating structured queries fromnatural language using reinforcement learning (Victor Zhong, Caiming Xiong,Richard Socher. CoRR2017)

[2] Towards Complex Text-to-SQL in Cross-DomainDatabase with Intermediate Representation (Jiaqi Guo, Zecheng Zhan, Yan Gao,Yan Xiao, Jian-Guang Lou, Ting Liu, and Dongmei Zhang. ACL2019)

[3] SParC: Cross-Domain Semantic Parsing in Context(Tao Yu, Rui Zhang, Michihiro Yasunaga, Yi Chern Tan, Xi Victoria Lin, Suyi Li,Heyang Er, Irene Li, Bo Pang, Tao Chen, Emily Ji, Shreya Dixit, David Proctor,Sungrok Shim, Jonathan Kraft, Vincent Zhang, Caiming Xiong, Richard Socher,Dragomir Radev. ACL2019)

[4] Pointer Networks（OriolVinyals, Meire Fortunato, Navdeep Jaitly. NIPS2015）

[5] SQLNet: Generating Structured Queries FromNatural Language Without Reinforcement Learning (Xiaojun Xu, Chang Liu, DawnSong. ICLR2018)

[6] TypeSQL: Knowledge-based Type-Aware NeuralText-to-SQL Generation (Tao Yu, Zifan Li, Zilin Zhang, Rui Zhang, DragomirRadev. NAACL2018)

[7] SyntaxSQLNet: Syntax Tree Networks for Complexand Cross-DomainT ext-to-SQL Task. (Tao Yu, Michihiro Yasunaga, Kai Yang, RuiZhang, Dongxu Wang, Zifan Li, and Dragomir Radev. EMNLP2018)

[8] Global Reasoning over Database Structures forText-to-SQL Parsing (Ben Bogin, Matt Gardner, Jonathan Berant. EMNLP2019)

[9] RAT-SQL: Relation-Aware Schema Encoding andLinking for Text-to-SQL Parsers (Bailin Wang, Richard Shin, Xiaodong Liu,Oleksandr Polozov, Matthew Richardson. Submitted to ACL2020)

[10] A Pilot Study for Chinese SQL Semantic Parsing(Qingkai Min , Yuefeng Shi and Yue Zhang EMNLP2019)