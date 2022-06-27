import numpy as np

bos_index = 0
eos_index = 2
pad_index = 1
ignore_index = -100
mask_index = 4
mask_percent = 0.15

sequence_length = 48
vocab_length = 767
samples = np.load("/g/g92/tran71/tran71/lbann_new/applications/nlp/Roberta_zinc_base/zinc250k.npy", allow_pickle=True) 

train_samples = samples[:int(samples.size*0.8)]
val_samples = samples[int(samples.size*0.8):int(samples.size*0.9)]
test_samples = samples[int(samples.size*0.9):]


# Masking samples
def masking(sample):
    sample_masked = sample.copy()
    rand = np.random.uniform(size=(1,sequence_length))
    replace = (rand < mask_percent) * (sample != bos_index) * (sample != eos_index) * (sample != pad_index) 
    mask_idx = np.nonzero(replace)[1]
    for idx in mask_idx:
        chance = np.random.uniform()
        if(chance < 0.1): #replace with random character excluding special characters
            sample_masked[idx] = np.random.randint(5,vocab_length) 
        elif (0.1 < chance < 0.9): #replace with mask character
            sample_masked[idx] = mask_index 
    return sample_masked,mask_idx

# Train sample access functions
def get_train_sample(index):
    sample = train_samples[index]
    if len(sample) < sequence_length:
        sample = np.concatenate((sample, np.full(sequence_length-len(sample), pad_index)))
    else:
        sample = np.resize(sample, sequence_length)

    sample_mask, mask_idx = masking(sample)

    idx = [i for i in range(0,sequence_length)]
    non_mask_idx = [i for i in idx if (i not in mask_idx)]

    label  = sample.copy()

    label[non_mask_idx] = ignore_index

    sample_all = np.full(3*sequence_length, pad_index, dtype=int)
    sample_all[0:len(sample)] = sample
    sample_all[sequence_length:2*sequence_length] = sample_mask
    sample_all[2*sequence_length:3*sequence_length] = label

    return sample_all


# Validation sample access functions
def get_val_sample(index):
    sample = val_samples[index]
    if len(sample) < sequence_length:
        sample = np.concatenate((sample, np.full(sequence_length-len(sample), pad_index)))
    else:
        sample = np.resize(sample, sequence_length)

    mask_idx = np.random.randint(0,sequence_length)
    #print(mask_idx)

    sample_mask = sample.copy()
    sample_mask[mask_idx] = 14

    idx = [i for i in range(0,sequence_length)]
    non_mask_idx = [i for i in idx if (i != mask_idx)]
    #print(non_mask_idx)

    label  = sample.copy()

    label[non_mask_idx] = ignore_index

    sample_all = np.full(3*sequence_length, pad_index, dtype=int)
    sample_all[0:len(sample)] = sample
    sample_all[sequence_length:2*sequence_length] = sample_mask
    sample_all[2*sequence_length:3*sequence_length] = label

    return sample_all


# Test sample access functions
def get_test_sample(index):
    sample = test_samples[index]
    if len(sample) < sequence_length:
        sample = np.concatenate((sample, np.full(sequence_length-len(sample), pad_index)))
    else:
        sample = np.resize(sample, sequence_length)

    mask_idx = np.random.randint(0,sequence_length)

    sample_mask = sample.copy()
    sample_mask[mask_idx] = 14

    idx = [i for i in range(0,sequence_length)]
    non_mask_idx = [i for i in idx if (i != mask_idx)]

    label  = sample.copy()

    label[non_mask_idx] = ignore_index

    sample_all = np.full(3*sequence_length, pad_index, dtype=int)
    sample_all[0:len(sample)] = sample
    sample_all[sequence_length:2*sequence_length] = sample_mask
    sample_all[2*sequence_length:3*sequence_length] = label

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
    return 767


