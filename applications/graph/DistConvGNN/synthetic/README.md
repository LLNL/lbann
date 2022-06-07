# Experiments with Distributed Graph Convolutions

## Dependencies

- NVSHMEM
- DistConv

Must build LBANN with `+nvshmem` and `+distconv`. 

Set the following environment flags prior to running: 

```
LBANN_KEEP_ERROR_SIGNAL=1
LBANN_HAS_NVSHMEM=1
```

## Running NVSHMEM-based Distributed  Scatter/Gather


Run either `DistGather.py` and `DistScatter.py`  with:

```
python3 DistScatter(Gather).py --num-rows <int> --num-cols <int> --out-rows <int> --enable-distconv 
```

## Running Distributed GCN

Run `GCN-DistConv.py` with: 

```
python3 GCN-DistConv.py --num-vertices <int> --num-features <int> --num-edges <int> --enable-distconv 
```
