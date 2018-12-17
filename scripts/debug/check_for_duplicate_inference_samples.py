# Checks a directory of NumPy files to see if there are any duplicates in the output files
# usage: python check_for_duplicate_inference_samples.py <path_to_directory_with_npy_files>
import numpy as np
import sys
import os

base = sys.argv[1]
os.system('ls ' + base + '/*.npy > filelist')
f = open('filelist').readlines()

keepme = None
m = []
print 'loading and concatenating numpy arrays ...'
for line in f :
  a = np.load(line[:-1])
  keepme = a
  m.append(a)
c = np.concatenate(m)

print '======================================================='
print 'stats for concatenated numpy array:'
size_1 = c.size
print 'size: ', c.size
print 'ndim: ', c.ndim
print 'shape:', c.shape
#print 'first entry in concatenated array:', c[0]

(array, indices, counts) = np.unique(c, return_index=True, return_counts=True, axis=0)

print
size_2 = array.size
print 'stats for unique-ified array:'
print 'size: ', array.size
print 'ndim: ', array.ndim
print 'shape:', array.shape
#print 'first entry in unique-ified array:', array[0]

print
print 'BOTTOM LINE: do sizes match?', (size_1 == size_2)
print "====================================================================="
# print 'sanity test:'
# cc = np.concatenate((c, keepme))
# (array, indices, counts) = np.unique(cc, return_index=True, return_counts=True, axis=0)

# size_1 = cc.size
# print 'stats for concatenated numpy array, constructed with duplicates:'
# print 'ndim: ', cc.ndim
# print 'shape:', cc.shape
# print 'size: ', cc.size

# size_2 = array.size
# print 'stats for unique-ified array:'
# print 'ndim: ', array.ndim
# print 'shape:', array.shape
# print 'size: ', array.size
# #print 'first entry in unique-ified array:', array[0]
# print
# print 'BOTTOM LINE: do sizes match?', (size_1 == size_2)
