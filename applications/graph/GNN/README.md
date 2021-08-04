## LBANNs Implementation of Graph Convolutional Kernels 
This directory contains models which use graph convolution kernels. The graph sub-module in lbann.modules enables 
geometric deep learning on LBANN. 

## Datasets
The datasets used to test the graph layers are: 

1. MNIST Superpixel 
2. PROTEINS
3. OGB

To automatically download the MNIST Superpixel dataset: 

```
cd data/MNIST_Superpixel
python3 MNIST_Superpixel_Dataset.py
```

To add self loops and normalize the adjacency matrix, run: 

```
python3 update_adj_mat.py
```

To automatically download the PROTEINS dataset: 
```
cd data/PROTEINS
python3 PROTEINS_Dataset.py
```

Note: Both datasets require significant amount of preprocessing post download, so 
the download and processing step should be run using the scheduler. 


## Running Instructions 
To run the a model with a graph kernel and a dataset: 

```
python3 main.py --dataset (Proteins/MNIST) --model (GCN/GIN/GRAPH/GATEDGRAPH) --mini-batch-size MB --num-epochs N

```

## Edge Conditioned Neural Networks 

## Links 

- Li, Yujia, et al. "Gated graph sequence neural networks." arXiv preprint arXiv:1511.05493 (2015).
- Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).
- Xu, Keyulu, et al. "How powerful are graph neural networks?." arXiv preprint arXiv:1810.00826 (2018).
- Morris, Christopher, et al. "Weisfeiler and leman go neural: Higher-order graph neural networks." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. 2019.
