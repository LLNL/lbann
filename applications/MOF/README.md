# Example models for 3D molecular generation 

This directory contains LBANN implementations of 3D molecular generation models for Metal Organic Frameworks from the CoRE MOF Database. The models are based on 3D convolutional fliters on periodic 3D voxel grids. 


## Dataset Information 

The dataset used is a subset of the [CoRE MOF Database](https://gregchung.github.io/CoRE-MOFs/). Each Metal Organic Framework is represented as a 32x32x32x11 tensor. 

The representation is channel-wise concatenation of 11 32x32x32 voxel grids, where each voxel grid represents the location of a particular element. 


## Links  

For more information on the data representation: 

## Running Instructions

Run the 
```
python3 MOFae.py --nodes N  --procs-per-node P --mini-batch-size B 
```


@article {Kimeaax9324,
	author = {Kim, Baekjun and Lee, Sangwon and Kim, Jihan},
	title = {Inverse design of porous materials using artificial neural networks},
	volume = {6},
	number = {1},
	elocation-id = {eaax9324},
	year = {2020},
	doi = {10.1126/sciadv.aax9324},
	publisher = {American Association for the Advancement of Science},
	}
	eprint = {https://advances.sciencemag.org/content/6/1/eaax9324.full.pdf},
	journal = {Science Advances}
}
 
