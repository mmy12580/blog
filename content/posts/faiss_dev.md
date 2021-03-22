---
title: "Industrial Solution: FAISS" 
date: 2019-03-02T01:55:11-05:00
tag: ['deployment', 'machine learning']
category: ['machine learning']
---

# Intuition 

Imagine a case, you are developing a facial recognition algorithm for Canadian Custom, and they would like to use it to identify different scenarios, e.g., criminals. Precision and speed of your models are higly demanded. Let us assume you have already tried your best to provide a promising performence of identifying every visitor, however, due to it is a trained on vary large database (40 million population in Canada), searching an image over the huge databse can be very time-consuming, so, what can we do?


# Faiss

My solution is to use a powerful tool created by Facebook named as **Faiss**. If you are a nlper, you should have used it already, but you may not know when and what you used :smile:. No worries. I am going to explain it to you soon. 

Before we introduce Faiss for sovling the Canadian Custom cases, now let us look at a real and simpler case. When you build a word embedding system, if you would like to find the most similar 10 words to a given word, say, **sushi**, what do you normally do?

### Numpy Users. 

`np.memmap` is a good trick to use when you have a large word embeddings to load to the memory since they can be shared and reloaded based on the limits of usage in RAM.

```
import numpy as np 
# define cosine distance
def cosine_distance(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    return vec1.dot(vec2) / (vec1_norm * vec2_norm)

# loading embeddings via np.memmap
embeds = np.memmap('wiki200', dtype='float32', model=r)
results = []
for item in embeds:
    eword, embed = item
    dist = cosine_distance(words, embed)
    results.append(eword, dist)

# sort results
print(sorted(results, key=lambda x: x[1])[:10])
```

### Gensim User. 

A more advanced and convenient tool for nlpers now is to use Gensim, a popular Python package. The key of Gensim to retrieve the most similar words for the query word is to use [Annoy](https://github.com/spotify/annoy), which creates large read-only file-based data sturctures that are mmapped into memory so that many processes may share the same data.
```
from gensim.similarities.index import AnnoyIndexer
from gensim.models import KeyedVectors

# load pretrained model 
model = KeyedVectors.load('wiki200.vec', binary=False)
indexer = AnnoyInnder(model, num_trees=2)

# retrieve most smiliar words
mode.most_similar('sushi', topn=10, indexer=indexer)
[('sushi', 1.0), 
 ('sashimi', 0.88),
 ('maki', 0.81),
 ('katsu', 0.64 )]
```

### Any thing better?

Both methods work in some cirumstances, nevertheless, it does not provide satistifactory results in production sometimes, especially large scalable cases. What other tools are available then? Remember the usuage of FastText on searching nearest words given query?

```
./fasttext nn wiki200.bin 
Query word? sushi
sushi 1.0
sashimi 0.88
maki 0.81
katsu 0.64
```

It is really fast, and yes, the algorithm behind is the super poweful tool, named as **Faiss**. It is  what we also apply at *Leafy.ai*. Of course, there are other alternatives, but I will only include Faiss here.

Basically，**Faiss** takes the encoded data as vectors (embedding e.g., images or texts), and then apply clustering and quantization algorithms to build an efficient index-based model. Does that sound like normal machine learning algorithms? Yes, they essentially do but are they able to handle millions of images, texts or even billions? Faiss based index model is written in C++ and highly optimized by 

1. Better multi-processing searching on CPU/GPU;
2. **BLAS** powered matrix calculation;
3. **SIMD** and **popcount** based fast distance calculation 

to solve those hard problems.


### Evaluation

As I just introduced, Faiss is designed to apply fast nearest neighbor search, so how good is it? Here, Facebook quantifies it by three direcitons

1. Speed. How long does it take to find the top 10 closest items?
2. RAM usage. How much RAM is required? This is directly relate to what machine is available for deployment.
3. Accuracy/Precision. 10-intersection is applied.


**Luckily**, Faiss provides `autotune` to search over parameter spaces. **It guarantees that given the accuracy requirement, they will find the best potential time to search, and vice versa.** One of the autotune examples in the [benchmark tests](https://github.com/facebookresearch/faiss/blob/master/benchs/README.md) is on 1 billion data [Deep1B](https://yadi.sk/d/11eDCm7Dsn9GA) shown as followings:

![](/post_imgs/deep1b.jpg)

As we can see from the plot that retrieving "required" 40% 1-recall@1 takes less than 2ms per vector indexing only. If we want even faster speed of search, e.g., 0.5 ms, we can still ahve 30%。In conclusion, we can have 500 queries per second over a single thread by 2ms.


Another benchmark test, clustering n=1M points in d=256 dimensions to k=20000 centroids (niter=25 EM iterations) is a brute-force operation that costs n * d * k * niter multiply-add operations, 128 Tflop in this case. The Faiss implementation takes:

- 11 min on CPU
- 3 min on 1 Kepler-class K40m GPU
- 111 sec on 1 Maxwell-class Titan X GPU
- 55 sec on 1 Pascal-class P100 GPU (float32 math)
- 52 sec on 4 Kepler-class K40m GPUs
- 35 sec on 4 Maxwell-class Titan X GPUs
- 34 sec on 1 Pascal-class P100 GPU (float16 math)
- 21 sec on 8 Maxwell-class Titan X GPUs
- 21 sec on 4 Pascal-class P100 GPUs (float32 math)
- 16 sec on 4 Pascal-class P100 GPUs (float16 math) *** (majority of time on the CPU)
- 14 sec on 8 Pascal-class P100 GPUs (float32 math) *** (problem size too small, majority of time on the CPU and PCIe transfers!)
- 14 sec on 8 Pascal-class P100 GPUs (float16 math) *** (problem size too small, bottlenecked on GPU by too small Hgemm size, majority of time on the CPU and PCIe transfers!)


At Leafy.ai, we have more than 10Million word embeddings, and 300K articles (max_seq_len = 8000 tokens) for searching, and Faiss provides promising precision and speed to solve our needs not only severing online but also normal smart phones. 


## How can we use it ?

### 1. Get your data ready first

They only need two things from data, **database** and **query vectors**. Apparently, database is to provide the pool for your query, and query vectors are queries asked to find the nearest neighbours. A toy example is like below:

```
import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32') # databse 
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32') # query vectors
xq[:, 0] += np.arange(nq) / 1000.
```

Note that dtype has to be float32 in Python numpy array.


### 2. Select index model

Faiss provides different index functions, and below is most commonly applicable cases.


| Method                                  | Class name      | index_factory | Main parameters | Bytes/vector | Exhaustive |
|--------------------------------------------|-----------------|-----------------|-----------------|--------------|------------|
| Exact Search for L2                        | IndexFlatL2   | "Flat"        | d             |4*d             | yes |
| Exact Search for Inner Product             | IndexFlatIP   | "Flat"        | d        | 4*d             | yes |
| Hierarchical Navigable Small World graph exploration | IndexHNSWFlat | "HNSWs, Flat" | d, M | 4*d + 8*M  | no | 
| Inverted file with exact post-verification | IndexIVFFlat  | "IVFx,Flat"   | quantizer, d, nlists, metric| 4*d | no |
| Locality-Sensitive Hashing (binary flat index) | IndexLSH  |             |d, nbits | nbits/8 | yes |
| Scalar quantizer (SQ) in flat mode          | IndexScalarQuantizer | "SQ8"   | d       | d              |  yes |
| Product quantizer (PQ) in flat mode         | IndexPQ       | "PQx"          |d, M, nbits | M (if nbits=8) | yes |
| IVF and scalar quantizer | IndexIVFScalarQuantizer | "IVFx,SQ4" "IVFx,SQ8" | quantizer, d, nlists, qtype | SQfp16: 2*d, SQ8: d or SQ4: d/2 | no |
| IVFADC (coarse quantizer+PQ on residuals)  | IndexIVFPQ | "IVFx,PQy"       | quantizer, d, nlists, M, nbits | M+4 or M+8 | no |
| IVFADC+R (same as IVFADC with re-ranking based on codes) |IndexIVFPQR | "IVFx,PQy+z"         |quantizer, d, nlists, M, nbits, M_refine, nbits_refine | M+M_refine+4 or M+M_refine+8  | no |

#### Fast Guideline!

They also provide a guideline for choosing an index given caraties of scenarios.

- If acucracy matters, then "Flat"
- Is memory an issue:
	* No: "HNSWx"
	* Somewhat, then "..., Flat"
	* Quite important, then "PCARx,...,SQ8"
	* Very important, then "OPQx_y,...,PQX"
- Data size:
	* Less than 1M vectors: "...,IVFx,..."
	* 1M - 10M: "...,IVF65536_HNSW32,..."
	* 10M - 100M: "...,IVF262144_HNSW32,..."
	* 100M - 1B: "...,IVF1048576_HNSW32,..."

#### AutoTune

As mentioned earlier, AutoTune can be really helpful for finding the promising index. AutoTune is mainly perfored on the running parameters.

| key | Index class | runtime parameter  | comments 
| ----|-------|--------------------|----------
| IVF*, IMI2x* | IndexIVF* | nprobe | the main parameter to adjust the speed-precision tradeoff
| IMI2x* | IndexIVF | max_codes | useful for the IMI, that often has unbalanced inverted lists
| PQ* | IndexIVFPQ, IndexPQ | ht | Hamming threshold for polysemous 
| PQ*+* | IndexIVFPQR | k_factor | determines how many result vectors are verified


### 3. General code in Python

After selecting index, then we can do some general process for building index and some I/O works. Recall the data preparation part, we have database `xb` and the query vectors `xq`.

```
# Define a quantizer and index 
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist)
assert not index.is_trained
index.train(xb)
assert index.is_trained # this matters if you are using a quantizer

index.add(xb)                  # add may be a bit slower as well
D, I = index.search(xq, k)     # actual search
print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more
D, I = index.search(xq, k)
print(I[-5:])                  # neighbors of the 5 last queries

# Extra if GPU wanted
### Single GPU
res = faiss.StandardGpuResources()  
gpu_index = faiss.index_cpu_to_gpu(res, 0, index) # CPU version from above

### Multiple GPUs
ngpus = faiss.get_num_gpus()
multi_gpu_index = faiss.index_cpu_to_all_gpus(index)

# I/O functions
### Save to the local file 
write_index(index, "large.index")

### Read from saved files
new_index = read_index("large.index")
```

Results are like this:

```
[[ 9900 10500  9831 10808]
 [11055 10812 11321 10260]
 [11353 10164 10719 11013]
 [10571 10203 10793 10952]
 [ 9582 10304  9622  9229]]
```

## Not enough for you?

Check [this](https://github.com/facebookresearch/faiss/blob/master/demos/demo_auto_tune.py) and make it deployable to your own applications.