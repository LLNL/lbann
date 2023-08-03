import numpy as np

bos_index = 12
eos_index = 13
pad_index = 0
ignore_index = -100
mask_index = 14
mask_percent = 0.15

sequence_length = 57
vocab_length = 600
samples = np.load("/g/g92/tran71/ChemBERTa-77M-MLM_smile2M.npy", allow_pickle=True) 


train_samples = samples[:int(samples.size*0.8)]
val_samples = samples[int(samples.size*0.8):int(samples.size*0.9)]
test_samples = samples[int(samples.size*0.9):]


# Masking samples
'''
https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/data/data_collator.py#L805
'''
def masking(sample, mlm_probability = 0.15):

  masked = np.copy(sample)
  label = np.copy(sample) 

  special_tokens_mask = (sample == bos_index) + (sample == eos_index) + (sample == pad_index) 
  
  probability_matrix = np.full(sample.shape, mlm_probability)

  probability_matrix[special_tokens_mask] = 0

  masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)

  label[~masked_indices] = ignore_index

  indices_replaced = np.random.binomial(1, 0.8, size=sample.shape).astype(bool) & masked_indices

  masked[indices_replaced] = mask_index

  indices_random = (np.random.binomial(1, 0.5, size=sample.shape).astype(bool) & masked_indices & ~indices_replaced)

  random_words = np.random.randint(low=5, high=vocab_length, size=np.count_nonzero(indices_random), dtype=np.int64)

  masked[indices_random] = random_words

  return sample,masked,label

# Train sample access functions
def get_train_sample(index):
    sample = train_samples[index]
    if len(sample) < sequence_length:
        sample = np.concatenate((sample, np.full(sequence_length-len(sample), pad_index)))
    else:
        sample = np.resize(sample, sequence_length)

    sample,masked,label = masking(sample)

    sample_all = np.full(3*sequence_length, pad_index, dtype=int)
    sample_all[0:len(sample)] = sample
    sample_all[sequence_length:2*sequence_length] = masked
    sample_all[2*sequence_length:3*sequence_length] = label

    return sample_all


# Validation sample access functions
def get_val_sample(index):
    sample = val_samples[index]
    if len(sample) < sequence_length:
        sample = np.concatenate((sample, np.full(sequence_length-len(sample), pad_index)))
    else:
        sample = np.resize(sample, sequence_length)

    sample,masked,label = masking(sample)

    sample_all = np.full(3*sequence_length, pad_index, dtype=int)
    sample_all[0:len(sample)] = sample
    sample_all[sequence_length:2*sequence_length] = masked
    sample_all[2*sequence_length:3*sequence_length] = label

    return sample_all

# Testing sample access functions
def get_test_sample(index):
    sample = test_samples[index]
    rand_size = np.random.randint(1, 4)

    if len(sample) < sequence_length:
        mask_idx = sorted(np.random.randint(1,len(sample)-1,size=rand_size))
    else:
        mask_idx = sorted(np.random.randint(1,sequence_length-1,size=rand_size))


    sample_mask = sample.copy()

    op = np.random.randint(0, 3)

    if(op == 0): # Delete augmentation
        for idx in mask_idx:
            if(idx < len(sample_mask)-2): # Don't mask the last character for deletion.          
                sample_mask[idx] = mask_index   

        del_count = 0            
        for idx in mask_idx:
            if(idx < len(sample_mask)-1): # Don't delete EoS.
                sample_mask = np.delete(sample_mask, idx+1+del_count)
                del_count -= 1           
    elif(op == 1): # Insert augmentation
        for idx in mask_idx:
            sample_mask = np.insert(sample_mask, idx, mask_index)     
    else: # Replace augmentation
        for idx in mask_idx:
            sample_mask[idx] = mask_index    
            
    if len(sample) < sequence_length:
        sample = np.concatenate((sample, np.full(sequence_length-len(sample), pad_index)))

    else:
        sample = np.resize(sample, sequence_length)

    if len(sample_mask) < sequence_length:
        sample_mask = np.concatenate((sample_mask, np.full(sequence_length-len(sample_mask), pad_index)))

    else:
        sample_mask = np.resize(sample_mask, sequence_length)

    sample_all = np.full(3*sequence_length, pad_index, dtype=int)
    sample_all[0:len(sample)] = sample
    sample_all[sequence_length:2*sequence_length] = sample_mask
    sample_all[2*sequence_length:3*sequence_length] = sample

    return sample_all 

def num_train_samples():
    return train_samples.shape[0]

def num_val_samples():
    return val_samples.shape[0]

def num_test_samples():
    return val_samples.shape[0]

def sample_dims():
    return (3*sequence_length+1,)

def vocab_size():
    return vocab_length

