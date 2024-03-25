# ChemProp on LBANN

## Prepere Dataset (optional)

If not on lbann system or required to regenerate the data file so it is ingestible on LBANN. 

### Requirements

```
chemprop
numpy
torch
```

### Generate Data

The data generator is set to read from and write data to the `DATA_DIR` directory in `config.py`. Update that line to read and store
from a custom directory.


Generate the data by calling:


`python PrepareDataset.py
`

## Run the Trainer

### Hyperparameters

The hyperparameters for the model and training algorihms can be set in `config.py`.


### Run the trainer


### Results

