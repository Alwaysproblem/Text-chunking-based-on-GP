# Text-chunking-based-on-GP
<!-- this a lab for practicing the AutoGP and BGPLVM and SAVIGP -->

## Background

In this academic project, we made use of the dataset from _the CoNLL-2000 shared task_ on **text chunking** (Tjong Kim Sang and Buchholz, 2000). Text chunking is concerned with dividing the text into syntactically-related chunks of words, or phrases. These phrases are non-overlapping in the sense that a word can only be a member of one phrase. For example, consider the sentence:

_He reckons the current account deficit will narrow to only #1.8 billion in September._

The segmentation of this sentence into chunks and their corresponding labels is shown in table 1. The chunk label contains the type of the chunk, e.q. **I-NP** for noun phrase words and **I-VP** for verb phrase words. Most chunk types have two kinds of labels to delineate the boundaries of the chunk, **B-CHUNK** for the first word of the chunk and **I-CHUNK** for every other word in the chunk. While all the necessary information to carry out this assignment is contained within this assignment specification, you may also find out more about this task at [Here](https://www.clips.uantwerpen.be/conll2000/chunking/)

---------

Table 1.

|           |      |
|:---------:| ---- |
| He        | B-NP |
| reckons   | B-VP |
| the       | B-NP |
| current   | I-NP |
| account   | I-NP |
| deficit   | I-NP |
| will      | B-VP |
| narrow    | I-VP |
| to        | B-PP |
| only      | B-NP |
| \#        | I-NP |
| 1.8       | I-NP |
| billion   | I-NP |
| in        | B-PP |
| September | B-NP |
| .         | 0    |

-----------

## Data

Instead of providing with raw text data, we have preprocessed and extracted features from this dataset. These are given in the compressed file "**conll_train.zip**" and "**conll_test_features.zip**". When extracted, you will find less "_i.x_" and "_i.y_" consisting of the features and chunk labels for the *i*th sentence, respectively.

### Schema
Let $T_i$ be the length of the *i*th sentence, the number of words/tokens it contains. There is a $D$-dimensional binary feature vector for each word/token in the sentence, where $D$ = 2, 035, 523. Due to the high-dimensionality of the feature space, the “_i.x_” file provides a _sparse_ representation of the feature vectors for the *i*th sentence. A row entry with the value

> j k 1

indicates that the *k*th feature for the *j*th word/token in the sentence has value 1.  Next, the “_i.y_” file contains the label $c \in \{1, . . . , 23\}$ of each of the $T_i$ words/tokens in the sentence.


## Preprocessing

The input data can be convert to a very high dimention Sparse Matrix. So we represented the input words/token with Scipy sparse Matrix class and reduce the dimention to 200 with TruncatedSVD on scikit-learn.

## Model

### Sparse Variational Gaussian approximation

we pick up some inducing point with k-means method to reduce the consumption of the computational resource and to improve the model accuracy up to around 90%, which is obviosly prior to the softmax classifier (82%).

More Information on [**_SVGP_**](http://proceedings.mlr.press/v38/hensman15.pdf)

### Scalable Automated Variational Inference for Gaussian Process

I transplanted the SAVIGP on the [_**Sparse-GP**_](https://github.com/Alwaysproblem/Sparse-GP/) on python3 environment and fix the ARD bug in the kernel.py. I also turn on the pytorch cuda accerleration for SAVIGP model on my [Github](https://github.com/Alwaysproblem/SAVIGP)

More Information on [_**SAVIGP**_](https://arxiv.org/abs/1609.00577)


TODO:
- [ ] Need to modified the source code for fitting data and see how it works and fix the consumption of memory using tensorflow possibility or pytorch.

### AutoGP

More Information on [**_AutoGP_**](https://github.com/ebonilla/AutoGP)

TODO:
- [ ] Need to transplant to the python3.

## Reference
 - [1] Tjong Kim Sang, E. F. and Buchholz, S. (2000). Introduction to the conll-2000 shared task: Chunking. In _Proceedings of the 2nd workshop on Learning language in logic and the 4th conference on Computational natural language learning-Volume 7,_ pages 127–132. Association for Computational Linguistics.
