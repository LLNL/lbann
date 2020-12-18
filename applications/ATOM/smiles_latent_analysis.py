import numpy as np
import pandas as pd
import sys
import glob
import torch

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

from moses.script_utils import  read_smiles_csv

def get_smiles_from_lbann_tensors(fdir, sequence_length, zdim, batch_num=0):
  '''
    Converts LBANN output tensors to SMILES using rdkit library
    Each row of LBANN tensor is a concatenation of input, latent space and output data
    This function, converts the input and output tensor to equivalent SMILES string
    Save SMILES strings to fdir
  '''
  vocab_path = 'path/to/vocab/file/'
 
  out_files = glob.glob(fdir+"*epoch."+str(batch_num)+".step*conc*_output0*.csv")
  outs = np.loadtxt(out_files[0], delimiter=",")
  for i, f in enumerate(out_files):
    if(i > 0) :
      outs = np.concatenate((outs, np.loadtxt(f,delimiter=",")))

  num_cols = outs.shape[1]
  print("Num cols ", num_cols)
  num_samples = outs.shape[0]
  print("Num samples ", num_samples)

  vocab = torch.load(vocab_path)
            
  #stop at eos (<.....)
  samples = [vocab.ids2string(i_x).split('<')[0] for i_x in outs[:num_samples,0:sequence_length]]

  samples = pd.DataFrame(samples, columns=['SMILES'])
  print("Save gt files to " , fdir+"gt_epoch"+batch_num+"smiles.txt")
  samples.to_csv(fdir+"gt_batch"+batch_num+"smiles.txt", index=False)

  samples = [vocab.ids2string(i_x).split('<')[0] for i_x in outs[:num_samples,sequence_length+zdim:]]

  samples = pd.DataFrame(samples, columns=['SMILES'])
  print("Save pred files to " , fdir+"pred_epoch"+batch_num+"smiles.txt")
  samples.to_csv(fdir+"pred_batch"+batch_num+"smiles.txt", index=False)

def compare_decoded_to_original_smiles(orig_smiles, decoded_smiles, output_file=None):
    """
    Compare decoded to original SMILES strings and output a table of Tanimoto distances, along with
    binary flags for whether the strings are the same and whether the decoded string is valid SMILES.
    orig_smiles and decoded_smiles are lists or arrays of strings.
    If an output file name is provided, the table will be written to it as a CSV file.
    Returns the table as a DataFrame.

    """
    res_df = pd.DataFrame(dict(original=orig_smiles, decoded=decoded_smiles))
    is_valid = []
    is_same = []
    tani_dist = []
    accuracy = []
    count = 0
    data_size = len(orig_smiles)
    for row in res_df.itertuples():
        count = count + 1
        #compute char by char accuracy
        hit = 0
        for x, y in zip(row.original, row.decoded):
            if x == y:
                hit = hit+1
        accuracy.append((hit/len(row.original))*100)

        is_same.append(int(row.decoded == row.original))
        orig_mol = Chem.MolFromSmiles(row.original)
        if orig_mol is None:
          print("INVALID AT input ", count, " ", row.original)
          #Note, input may be invalid, if original SMILE string is truncated 
          is_valid.append('x')
          tani_dist.append(-1)
          continue
        dec_mol = Chem.MolFromSmiles(row.decoded)
        RDLogger.DisableLog('rdApp.*')
        if dec_mol is None:
            is_valid.append(0)
            tani_dist.append(1)
        else:
            is_valid.append(1)
            orig_fp = AllChem.GetMorganFingerprintAsBitVect(orig_mol, 2, 1024)
            dec_fp = AllChem.GetMorganFingerprintAsBitVect(dec_mol, 2, 1024)
            tani_sim = DataStructs.FingerprintSimilarity(orig_fp, dec_fp, metric=DataStructs.TanimotoSimilarity)
            tani_dist.append(1.0 - tani_sim)
    res_df['is_valid'] = is_valid
    res_df['is_same'] = is_same
    res_df['smile_accuracy'] = accuracy
    res_df['tanimoto_distance'] = tani_dist
    global_acc  = np.mean(np.array(accuracy))
    res_df['total_avg_accuracy'] = [global_acc]*len(accuracy)
    
    print("Mean global accuracy % ", global_acc)
    print("Validity % ", (is_valid.count(1)/data_size)*100)
    print("Same % ", (is_same.count(1)/data_size)*100)
    valid_tani_dist = [ t for t in tani_dist if t >= 0 ] 
    print("Average tanimoto ", np.mean(np.array(valid_tani_dist)))
    

    if output_file is not None:
        output_columns = ['original', 'decoded', 'is_valid', 'is_same', 'smile_accuracy','tanimoto_distance','total_avg_accuracy']
        res_df.to_csv(output_file, index=False, columns=output_columns)
    return(res_df)


fdir = sys.argv[1] #directory of LBANN tensor outputs
sd = sys.argv[3]   #tag for say different noise pertubation values

sequence_length = 102 #Max sequence lenght use in LBANN training (100+bos+eos)
zdim = 128 #latent space dimension
batch_num = 0 #use to control loading different batches of dump (default 0) 

get_smiles_from_lbann_tensors(fdir,sequence_length, zdim)

orig_file = read_smiles_csv(fdir+"gt_batch"+batch_num+"smiles.txt")
pred_file = read_smiles_csv(fdir+"pred_batch"+batch_num+"smiles.txt")
#diff_file = fdir+"diff_epoch"+epoch_num+"smiles.txt"
diff_file = fdir+"sd"+sd+"_smiles_metrics.csv"

print("Input/pred SMILES file sizes ", len(orig_file), " ", len(pred_file))

compare_decoded_to_original_smiles(orig_file, pred_file, diff_file)
print("Input/pred SMILES diff file saved to", diff_file)
