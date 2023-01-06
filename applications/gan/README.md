# GAN Examples

This directory contains two examples for training GANs with LBANN: one for training on MNIST digits and the other for training on general image data. The MNIST demo can be trained with:

```
python train_gan.py --nodes=1 --procs-per-node=1
```

The MNIST demo will download the dataset automatically, but the image demo requires the dataset to be passed as an additional argument: 

```
python train_gan.py --nodes=1 --procs-per-node=1 --data-path="/path/to/data/*.jpg"`
```

During training, LBANN will periodically save generator samples to disc. These can be visualized using:

```
python generate_samps.py
```