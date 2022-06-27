import numpy as np
import pandas as pd

from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

'''
Re-tokenize SMILE string using Huggingface tokenizers.

'''

model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

  
def detokenize(inp,vocab):
  '''
  Convert the tokenized zinc to SMILE strings 
  '''
  output = ""
  for i in inp: 
    token = list(vocab.keys())[list(vocab.values()).index(int(i))]
    if(token =='<eos>'):
      break
    if(token[0]!='<'):
      output = output+token

  return output


samples = np.load("moses_zinc_train250K.npy", allow_pickle=True) 

vocab_file = "vocab_train.txt"

vocab = pd.read_csv(vocab_file, delimiter=" ", header=None, quoting=3).to_dict()[0]
vocab = dict([(v,k) for k,v in vocab.items()])


samples = [detokenize(i_x,vocab) for i_x in samples] 

print(len(samples))

smiles_tokenized=[]

for s in samples:
    tokenize = tokenizer.encode(s)
    del tokenize[-2] # remove extra character before <eos>
    smiles_tokenized.append(tokenize)

print(len(smiles_tokenized))
    
outfile = "zinc250k.npy"

np.save(outfile, smiles_tokenized)
