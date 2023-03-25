## K-FAC Optimization Method Examples

Sample models using the K-FAC callback.
This directory includes a 3-layer MLP and a 2-layer CNN on the MNIST classification dataset.
These models are compatible with the MNIST example models of the Chainer implementation
([tyohei/chainerkfac](https://github.com/tyohei/chainerkfac)).

### Reference
```
Martens, James and Roger Grosse. "Optimizing neural networks with
kronecker-factored approximate curvature." International conference
on machine learning. 2015.

Grosse, Roger, and James Martens. "A kronecker-factored approximate
fisher matrix for convolution layers." International Conference on
Machine Learning. 2016.

Osawa, Kazuki, et al. "Large-scale distributed second-order
optimization using kronecker-factored approximate curvature for
deep convolutional neural networks." Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition. 2019.
```

### How to Train

* Adam:
```bash
python3 kfac.py \
	--nodes 1 --procs-per-node 4 \
	--model mlp \
	--optimizer adam --optimizer-learning-rate 1e-4
```

* K-FAC:
```bash
python3 kfac.py \
	--nodes 1 --procs-per-node 4 \
	--model mlp \
	--optimizer sgd --optimizer-learning-rate 1e-1 \
	--kfac \
	--kfac-damping-act 0.01 --kfac-damping-err 1
```

Expected training output in LBANN is shown:

* Adam:
```
--------------------------------------------------------------------------------
[19] Epoch : stats formated [tr/v/te] iter/epoch = [600/0/100]
            global MB = [ 100/   0/ 100] global last MB = [ 100  /   0  / 100  ]
             local MB = [ 100/   0/ 100]  local last MB = [ 100+0/   0+0/ 100+0]
--------------------------------------------------------------------------------
Model 0 GPU memory usage statistics : 0.902 GiB mean, 0.902 GiB median, 0.902 GiB max, 0.902 GiB min (15.8 GiB total)
model0 (instance 0) training epoch 19 objective function : 0.00160377
model0 (instance 0) training epoch 19 accuracy : 99.9717%
model0 (instance 0) training epoch 19 run time : 0.80641s
model0 (instance 0) training epoch 19 mini-batch time statistics : 0.00132546s mean, 0.00162332s max, 0.00127891s min, 2.12701e-05s stdev
model0 (instance 0) test objective function : 0.075789
model0 (instance 0) test accuracy : 98.23%
model0 (instance 0) test run time : 0.0664401s
model0 (instance 0) test mini-batch time statistics : 0.000647512s mean, 0.00138687s max, 0.000630064s min, 7.48268e-05s stdev
```

* K-FAC:
```
--------------------------------------------------------------------------------
[19] Epoch : stats formated [tr/v/te] iter/epoch = [600/0/100]
            global MB = [ 100/   0/ 100] global last MB = [ 100  /   0  / 100  ]
             local MB = [ 100/   0/ 100]  local last MB = [ 100+0/   0+0/ 100+0]
--------------------------------------------------------------------------------
Model 0 GPU memory usage statistics : 0.969 GiB mean, 0.969 GiB median, 0.969 GiB max, 0.969 GiB min (15.8 GiB total)
model0 (instance 0) training epoch 19 objective function : 0.000145131
model0 (instance 0) training epoch 19 accuracy : 100%
model0 (instance 0) training epoch 19 run time : 7.60849s
model0 (instance 0) training epoch 19 mini-batch time statistics : 0.0126484s mean, 0.0152339s max, 0.00488174s min, 0.000337816s stdev
K-FAC callback: changing damping value to 0.01 (act), 1 (err), 0.03 (bn_act), 0.03 (bn_err) at 20 epochs
model0 (instance 0) test objective function : 0.070923
model0 (instance 0) test accuracy : 98.2%
model0 (instance 0) test run time : 0.0747263s
model0 (instance 0) test mini-batch time statistics : 0.000730126s mean, 0.00950197s max, 0.000634307s min, 0.000886054s stdev
```
