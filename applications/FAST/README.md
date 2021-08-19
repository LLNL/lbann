# Fusion model for Atomic and molecular STrcutre (FAST) - LBANN

This is the LBANN implementation of the FAST model using LBANN 

### Prerequisites 

The simulated data does not add additional requirements on top of LBANN and Numpy. 

The pre-processed PDBBind data prequisites are: 

- pyh5 
- pybel 

### Data Description 

#### Grid-Structured Data 

#### Graph-Strucured Data 

- Node Features:  
- Edge Features:  
- Covalent Edge COO List
    - Covalent Edge Sources:  
    - Covalent Edge Targets: 
- Non-Covalent Edge COO List
    - Non-Covalent Edge Sources:
    - Non-Covalent Edge Targets:  
- Ligand Only: 
- Target 

### Running the models 

Each of the 3DCNN, SGCNN, and FAST models can be run seperately. 

Run the 3DCNN model with:
```
python3 run_3dcnn.py --num-epochs N --mini-batch-size B 
```

Run the 3DCNN model with:
```
python3 run_SGCNN.py --num-epochs N --mini-batch-size B 
```

Run the FAST model with:
```
python3 run_FAST.py --num-epochs N --mini-batch-size B 
```
### Citation 

Derek Jones, Hyojin Kim, Xiaohua Zhang, Adam Zemla, William D. Bennett, Dan Kirshner, Sergio Wong, Felice
Lightstone, and Jonathan E. Allen, "Improved Protein-ligand Binding Affinity Prediction with Structure-Based Deep Fusion Inference", arxiv 2020. 
