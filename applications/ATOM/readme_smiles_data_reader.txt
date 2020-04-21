export BASE=/usr/workspace/wsb/hysom/RUNME/SMILES


NOTES: 
  1) num-embeddings, embedding-dim and pad-index are specific to the 
     vocabulary file, which differs for examine and zinc
  2) similarly, sequence-length also differs

#
# example cmd line for all2018q1-2REAL680SMILES.smiles on pascal
# Note: if --num_samples=<int> is optional; if not given, the
# entire file will be used, which is 680,034,795 samples, however,
# this hasn't been tested, so for now please specify --num-samples
#
run python3 train_atom_char_rnn.py \
  --nodes=8 \
  --batch-size=1024 \
  --num-samples=1000000 \
  --num-embeddings=40 \
  --embedding-dim=40 \
  --pad-index=39 \
  --sequence-length=120 \
  --use-data-reader-prototext \
  --data-reader-prototext=$BASE/smiles_data_reader.prototext \
  --vocab=$BASE/data/vocab_examine.txt \
  |& tee out

#
# example cmd line for  dataset_v1.csv (zinc) on pascal
# (ensure you edit smiles_data_reader.prototext for proper filedir/name)
#
run python3 train_atom_char_rnn.py \
  --nodes=8 \
  --batch-size=1024 \
  --num-embeddings=30 \
  --embedding-dim=30 \
  --pad-index=29 \
  --sequence-length=60 \
  --use-data-reader-prototext \
  --data-reader-prototext=$BASE/smiles_data_reader.prototext \
  --vocab=$BASE/data/vocab_zinc.txt \
  |& tee out

