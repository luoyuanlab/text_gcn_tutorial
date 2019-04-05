# Text GCN Tutorial

This tutorial (currently under development) is based on the implementation of Text GCN in our paper:

Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19)


# Require

Python 2.7 or 3.6

Tensorflow >= 1.4.0

# Reproduing Results

1. Run `python remove_words.py ohsumed_3`

2. Run `python build_graph.py ohsumed_3`

3. Run `python train.py ohsumed_3`

# Visualizing Documents
Run `python tsne.py`

# Example input data
The Ohsumed corpus is from the MEDLINE database, which is a bibliographic database of important medical literature maintained by the National Library of Medicine

In this tutorial, we created a subsample of the 2,762 unique diseases abstracts from 3 categories
C04: Neoplasms
C10: Nervous System Diseases
C14: Cardiovascular Diseases

As we focus on single-label text classification, the documents belonging to multiple categories are excluded

1230 train (use 10% as validation), 1532 test

1. `/data/ohsumed_3.txt` indicates document names, training/test split, document labels. Each line is for a document.

2. `/data/corpus/ohsumed_3.txt` contains raw text of each document, each line is for the corresponding line in `/data/ohsumed_3.txt`
