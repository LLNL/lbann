import os
import sys
import random
import numpy as np


p0_thresh = 0.55
p1_thresh = 0.85
p2_thresh = 0.85

def preprocess_data(dirspath,channels=None):
# define a tuple of specific channels if user listed them
  channels_tuple = tuple(range(14))
  if channels is not None:
    channels_tuple = tuple(channels)

  files_train = []
  states      = []
  cons      = []

  #for d in dirspath:
  for _ in range(1):
  # get list of all files in datapath and shuffle them
  # sort by filename before shuffle so we could generate
  # a consistent list if using the same random seed
    filenames = os.listdir(dirspath)
    filenames.sort()
    random.shuffle(filenames)
  
    filenames_divide  = int(1.0 * len(filenames))
    filenames_train = filenames[:filenames_divide]

    files_train.append([dirspath + "/" + f for f in filenames_train])
  
    frame_start = 0
  
    for f in filenames_train:
    # read in the data file
      d = np.load(dirspath + '/' + f)
  
    # extract fields
      p = d['probs'][d['frames'] >= frame_start]
      s = d['states'][d['frames'] >= frame_start]
      #n = d['density_sig1p5'][d['frames'] >= frame_start]
      n = d['density_sig1'][d['frames'] >= frame_start]
      #print p.shape, s.shape

      s = s[(p[:,0] > p0_thresh) | (p[:,1] > p1_thresh) | (p[:,2] > p2_thresh)]
      n = n[(p[:,0] > p0_thresh) | (p[:,1] > p1_thresh) | (p[:,2] > p2_thresh)]
  
      states.append(s)
  
  
      # append concentrations, filter out by channel id(s) if given
      # can we do channel first here, transpose?, move axis?
      n = np.array(n)
      n = n.astype(np.float32)
      if channels:
        cons.append(n[:,:,:,channels_tuple])
      else:
        cons.append(n)
  

  states      = np.concatenate(states,axis=0)
  cons        = np.concatenate(cons,axis=0)

  # print list of unique state labels and number of each
  (values, cnt) = np.unique(states, return_counts=True)

  min_cnt = np.min(cnt)
  idx_0 = np.where(states == 0)
  idx_0 = idx_0[0][:min_cnt]
  idx_1 = np.where(states == 1)
  idx_1 = idx_1[0][:min_cnt]
  idx_2 = np.where(states == 2)
  idx_2 = idx_2[0][:min_cnt]
  ids = np.concatenate([idx_0, idx_1, idx_2], axis=0)
  states = states[ids]
  cons   = cons[ids]


  # normalize each concentration channel independently
  mins = cons.min(axis=(0,1,2), keepdims=True)
  maxs = cons.max(axis=(0,1,2), keepdims=True)

  cons      /= maxs
  labels      = states
   
  #transpose to NCHW
  cons = cons.transpose(0,3,1,2)

  X = cons.reshape(cons.shape[0],-1)
  y = labels.reshape(-1,1)
  Xy_data = np.hstack((X,y))
  return Xy_data
