## LBANNs Implementation of Graph Convolutional Kernels 
This directory contains models which use graph convolution kernels. The graph sub-module in lbann.modules enables 
geometric deep learning on LBANN. 

## Datasets
The datasets used to test the graph layers are: 


1. PROTEINS
2. OGB-PCQM4M-LSC


Note: Both datasets require significant amount of preprocessing post download, so 
the download and processing step should be run using the scheduler. 


## Running Instructions 
To run the a model with a graph kernel: 

```
python3 main.py --model (GCN/GIN/GRAPH/GATEDGRAPH) --mini-batch-size MB --num-epochs N

```

## Edge Conditioned Neural Networks 

To run the edge conditioned network for OGB-PCQM4M-LSC dataset

```
python3 OGB_LBANN_Trainer.py --mini-batch-size MB --num-epochs N --ps P

```

For P > 0, the graph kernel utilizes channelwise distributed tensors for the the graph kernel. 
 
## Links 

- Li, Yujia, et al. "Gated graph sequence neural networks." arXiv preprint arXiv:1511.05493 (2015).
- Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).
- Xu, Keyulu, et al. "How powerful are graph neural networks?." arXiv preprint arXiv:1810.00826 (2018).
- Morris, Christopher, et al. "Weisfeiler and leman go neural: Higher-order graph neural networks." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. 2019.
