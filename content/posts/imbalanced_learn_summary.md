---
title: "A quick summary for imbalanced data"
date: 2019-06-18T11:28:17-04:00
draft: false
tags: ['machine learning', 'data']
categories: ['machine learning']
---


Data imbalance occurs when the sample size in the data classes are unevenly distributed. Such situation is encountered in many applications in industry. Sometimes, it could be extremely imbalanced e.g click-through rate prediction, fraud detection, or cancer diagnosis etc. Most of machine learning techniques work well with balanced training data but they face challenges when the dataset classes are imbalanced. In such situation, classification methods tend to be biased towards the majority class. However, the interest of classification is normally the minority class. Sadly 😔, they are normally a short amount of data or low quality data. Therefore, learning from classification methods from imbalanced dataset can divide into two approaches: 

1. **data-level strategies**  
2. **algorithmic strategies** 


In this blog, I will show what problems caused the learning difficult, and their state-of-the-art solutions. 


## Why is it difficult?

Before introducing the summary of solutions about imbalanced data, let us look at what makes the imbalanced learning difficult? Given a series of research about imbalanced classification, there are mainly four types of problems: 

1. Most of the minority class samples happen to be in high-density majority class samples
2. There is a huge overlap between different class distributions
3. Data is noisy, especially minority data
4. Sparsity on minority data and small disjuncts situation


### Illustrations: 
![Case 1: minority samples show up in high-density majority samples](https://sci2s.ugr.es/sites/default/files/files/ComplementaryMaterial/imbalanced/04clover5z-800-7-30-BI.png)

![Case 2: overlap](/post_imgs/overlap.jpg)


![Case 4: small disjuncts](https://sci2s.ugr.es/sites/default/files/files/ComplementaryMaterial/imbalanced/custom_data_small_disjunct_3.png)


## Data-level Strategy

The most intuitive way is to re-sample the data to make them somehow 'balanced' because in this case, we can still perform normal machine learning techniques on them. There are generally three types methods:

1. <span style="color:blue">Down-sampling from majority class</span> e.g RUS, NearMiss, ENN, Tomeklink
2. <span style="color:blue">Over-sampling from minority class</span> e.g SMOTE, ADASYN, Borderline-SMOTE
3. <span style="color:blue">Hybrid method </span> e.g Smote + ENN

There are pros and cons from data-level strategy.

### Pros: 

1. Boost the performance of classifiers by removing some noise data
2. Down-sampling can remove some samples so it is helpful for faster computation


### Cons:

1. Re-sampling method is generally finding neighborhood samples from distances. <span style="color:red">Curse of dimensionality happens! </span>. It wont be helpful for large-scale data.
2. Unreasonable re-sampling caused by noise may not accurately capture the distribution information, thus, yields bad performance.
3. Not applicable to some complex dataset since distance metric is inapplicable

Data-level strategy can be easily achieved by using python package, [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/introduction.html), which you can build a pipeline just like scikit-learn interface.

````python
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline

X = data.data[idxs]
y = data.target[idxs]
y[y == majority_person] = 0
y[y == minority_person] = 1

classifier = ['3NN', neighbors.KNeighborsClassifier(3)]

samplers = [
    ['Standard', DummySampler()],
    ['ADASYN', ADASYN(random_state=RANDOM_STATE)],
    ['ROS', RandomOverSampler(random_state=RANDOM_STATE)],
    ['SMOTE', SMOTE(random_state=RANDOM_STATE)],
]

# create a pipeline with sampling methods
pipelines = [
    ['{}-{}'.format(sampler[0], classifier[0]),
     make_pipeline(sampler[1], classifier[1])]
    for sampler in samplers
]

# train
for name, pipeline in pipelines:
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for train, test in cv.split(X, y):
        probas_ = pipeline.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
````



## Algorithmic strategy


### Cost-sensitive learning
In stead of touching data, we can also work on algorithms. The most intuitive way is the <span style="color:blue"> cost-sensitive learning </span>. Due to the cost of mis-classifying minority class (our interest) is higher than the cost of mis-classifying majority class, so the easiest way is to use Tree based method e.g decision tree, random forest, boosting or SVM methods by setting their weights as something like {'majority': 1, 'minority': 10}. 

Cost-sensitive learning doest not increase model complexity and it is flexible to use to any type of classification cases as. binary or multi-class classification by setting weights for cost. However, it requires some prior knowledges to build the cost matrix, and it dost not guarantee to have the optimal performance. In addition, it cant generalize among different tasks since the cost is designed for a specific tasks. Last but not least, it dost not help mini-batch training. The gradient update of a network will easily push optimizer to local minima or saddle point, so it is not effective to learn a neural network.


### Ensemble learning

Another method that seems to be getting more and more popular for solving data imbalance is ensembles such as SMOTEBoost, SMOTEBagging, Easy Ensemble or BalanceCascade. As far as I observe from my work, ensemble learning seems to the currently best method to solve data imbalance case; nevertheless, it requires more computational power and time to implement, and it might lead to non-robust classifiers.


## Experience

1. Down-sampling: It is able to remove some noise and it is very fast to implement. <span style="color:blue">Random Downsampling</span> can be used in any situation, but it might be harmful for high imbalanced ratio cases. <span style="color:blue">NearMiss</span> is very sensitive to noisy data. To remove noise of data, you can try <span style="color:blue">tomeklink</span>, <span style="color:blue">AllKNN</span>.
2. Oversampling: It is very easy to overfit the data. <span style="color:blue">SMOTE</span> and <span style="color:blue">ADASYN</span> could be helpful for small data.
3. Hybrid sampling: Also helpful for small dataset
4. Cost-sensitive: It takes time to pre-determine the cost-matrix, and it might work well by good settings and work badly by bad settings. 
5. Bagging is normally better than Boosting based ensemble method.


If you are solving deep learning case, especially compute vision based projects. To spend 20 mins reading Kaimin He's [paper](chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://arxiv.org/pdf/1708.02002.pdf), you will benefit a lot, and it can be used in other applications such as [fraud detection dataset on Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1), and you can check this [github](https://github.com/Tony607/Focal_Loss_Keras) to have a practice with <span style="color:blue"> focal loss </span>  . 



