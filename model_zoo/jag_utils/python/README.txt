As always, run without cmd params for usage

select_samples.py 

  generates sample lists from a master index.txt; it's preferable 
  (because it's faster) to use lbann/model_zoo/jag_utils/select_samples

build_trainer_lists.py

  This is a wrapper that calls lbann/model_zoo/jag_utils/select_samples
  to generate a set of sample_list files.
