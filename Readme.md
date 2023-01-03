Interpretable Clustering
=====

This repository contains the code for [Interpretable Clustering on Dynamic Graphs with Recurrent Graph Neural Networks](https://arxiv.org/abs/2012.08740), AAAI 2021.

## Data

6 datasets were used in the paper:

- Simulate
- DBLP-E
- DBLP-3
- DBLP-5
- Brain
- Reddit

Update:
Simulate and DBLP-E datasets work well.
DBLP-3, DBLP-5, Brain, and Reddit datasets are from another paper and hard to verify the meaning of node features. Will replace it to new datasets soon.

## Requirements
  * Python 3,
  * PyTorch 1.2.0 or higher,
  * TensorFlow==1.11.0,
  * dgl-cu101,
  * dynamicgem,
  * networkx,
  * sklearn,
  * scipy,
  * keras==2.2.4,
  * numpy>=1.15.3

## Main file
  * main.ipynb


