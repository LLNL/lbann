import numpy as np
from SMILES_tokenizer import MolTokenizer
from data_utils import random_zero_array


def main():
    tokenizer = MolTokenizer("SMILES_vocab.json")
    tokenizer.load_vocab_file()
    with open("QM9_smiles.txt", 'r') as smiles_data:
      smiles_data = smiles_data.readlines()
      num_samples = len(smiles_data)
      max_length = 32

      tokenized_data = np.ones((num_samples, max_length)) * tokenizer.encode(tokenizer.pad_token) 
      tokenized_data[:, 0] = tokenizer.encode(tokenizer.sep_token)

      for i, smiles in enumerate(smiles_data, start=1):
        tokens = tokenizer.tokenize(smiles)
        tokens = random_zero_array(tokens, 0.15, tokenizer.encode(tokenizer.mask_token))
        tokenized_data[i, :len(tokens)] = tokens
        tokenized_data[i, len(tokens)] = tokenizer.encode(tokenizer.cls_token)

    np.save('QM9_Pretokenized.npy', tokenized_data)

if __name__ == '__main__':
    main()
