## Mesh Graph Networks

This example contains an LBANN implementation of mesh-based graph neural network (MeshGraphNet) with 
synthetically generated data. 
For more information about the model, refer to: T. Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks". ICLR'21.

---
### Running the example

The data-parallel model can be run with the synthetic data with: 

```bash
python Trainer.py --mini-batch-size <Global_MB_size> --num-epochs <number of epochs>
```
