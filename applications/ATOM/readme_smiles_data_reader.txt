# Example execution line for running with the smiles_data_reader

setenv BASE /usr/workspace/wsb/hysom/corona/applications/ATOM

run python3 train_atom_char_rnn_REV.py \
  --nodes=16             \
  --batch-size=16000     \
  --num-epochs=10        \
  --sequence-length=120  \
  --embedding-dim=40     \
  --num-embeddings=40    \
  --pad-index=38         \
  --num-samples=1000000  \
  --vocab=$BASE/data/vocab_enamine.txt \
  --data-reader-prototext=$BASE/smiles_data_reader.prototext \
 |& tee out

WARNING: at present, code assumes the input file is in csv format with
         tab delimiters

Optional arguments:

  --num-samples=<int> # If not given, uses all samples in the file

Notes:
  --sequence-length, --vocab, --num-embeddings, and --embedding-dim should 
  match the data set; vocabs for enamine and zinc are in applications/ATOM/data.
  For now, assume num-embeddings = embedding-dim = vocab.size(), and 
  pad-index= vocab.size()-2
  
  If --sequence-length is too short, portions of some samples will be discarded.
  The smiles_data_reader dtor prints any characters that were not
  found in the vocabulary, and the number of characters (if any) that
  were discarded (but note, statistics are only gathered for P_0)

WARNING (when running with the Python data_reader): 
   ensure that "--sequence-length" matches "max_seq_len" in the 
   json file; this is not error-checked (as of this writing). 
   Also ensure "--pad-index" matches the entry in the json file; 
   also not error checked.

