# RoBERTa

This directory contains and LBANN implementation of an optimized version of the
BERT model, RoBERTa. This implementation is based on and validated against the
[HuggingFace PyTorch RoBERTa
implementation](https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/roberta/modeling_roberta.py#L695).

## Dependencies

- PyTorch

## Usage

You will need to run the `get_model_config.py` script to download the model
configuration file and pretrained weights from the HuggingFace repository. By
default, the RoBERTa model will load pretrained weights provided in the
HuggingFace implementation. If you want to train the model from scratch,
without loading pretrained weights, then run `get_model_config.py
--no-weights`:

```bash
# Download config and pretrained weights
python3 get_model_config.py

# Download just config
python3 get_model_config.py --no-weights
```

The directory should now contain a `config.json` file and optionally,
`pytorch_model.bin` and `pretrained_weights/`. Modifying values in
`config.json` will change the parameters used to build the RoBERTa model.

An example of how to run the model is provided in `main.py` and a synthetic
dataset is provided in `dataset.py`. Run the example with:

```bash
python3 main.py --nodes 1 --procs-per-node 2 --time-limit 60 --partition pbatch --epochs 5 --mini-batch-size 10
```
