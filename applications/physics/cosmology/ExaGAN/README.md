## Generative Models for Cosmology - Understanding the Nature of the Universe at Exascale 

LBANN implementation of a number of generative models for cosmology. Please see [link](https://github.com/pzharrington/ExaGAN/) for original Keras implementation of code in this directory and other details. Also, see LBANN documentations on how to install, build and run LBANN code. 

### How to Train 
```bash
##Vanilla ExaGAN model
run python3 train_exagan.py
# Multi GAN model
run python3 train_multigan.py --procs-per-node=4 --nodes=4 --num-discblocks=4 --mini-batch-size=64 --compute-mse --num-epochs=4 --use-bn (--enable-subgrah) 
# Multi conditional GAN
run python3  train_conditional_multigan.py   --job-name cGAN    --nodes=4   --procs-per-node=4  --use-bn  --num-discblocks=4  --input-width=128  --num-epochs 56  --mini-batch-size=32  --compute-mse (--use-hdf5-reader) (--enable-subgraph)
```
