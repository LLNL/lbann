## Example Models for Graph Structured Chemistry Data 

## Dependencies 

- torch 
- numpy  

If you are on an AMD or Intel machine, use: 

```
pip3 install torch torchvision 
```

If you are on a Power machine, (Ray or Lassen), easiest way to install is in a spack environment: 

```
spack add pytorch@1.5.0 cuda_arch=sm
spack install 
```

Where, sm is the CUDA architecture. 60 for Ray and 70 for Lassen. 

## Data

To automatically get the data and preprocess, 

```
cd data/MNIST_Superpixel
python3 dataset.py
```


## Usage 

Run with: 

```
pythone main.py --nodes N --mini-batch-size B 
```
