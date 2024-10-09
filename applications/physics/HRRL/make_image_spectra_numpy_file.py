#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import h5py
import numpy as np
import os
from datetime import datetime
from sklearn import preprocessing as pp
from sklearn.preprocessing import FunctionTransformer
from sklearn import model_selection

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NOTES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Creates a .npy file with the images and spectra from the HRRL data set stored 
# here: /usr/WS2/hrrl/Data/PROBIES/SimBased/300x300/RawFiles.  Splits the data into
# testing, training, and validation sets, and saves each of these sets into a .npy file
# in the current directory.
#
# Each sample in the .npy file is a 90201x1 vector, where there are 90000 image data 
# points and 201 spectra data points, in that order.
#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SETUP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

folder = '/usr/WS2/hrrl/Data/PROBIES/SimBased/300x300/RawFiles'
files = os.listdir(folder)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LOAD DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# loop over files; load; append
out_array = []

for i in range(len(files)):	
    if "h5base" in files[i]: # check that the name contains h5base
        f = h5py.File(folder + '/' + files[i], 'r')
        rids = f['RUN_ID']
        for tid in rids:
            # check that we have the spectra
            if (rids[tid]['Original_Spectrum'].shape[0] == 201): 
                out_array.append(np.append(
                    np.array(rids[tid]['Image']).flatten(), 
                    rids[tid]['Original_Spectrum']
                    ))
            else:
                warning('no spectrum info for' + str(tid))

out_array = np.array(out_array, dtype=object)

values_raw = out_array[:,:-201]
labels_raw = out_array[:,-201:]

print("raw value array shape", values_raw.shape)
print("raw labels array shape", labels_raw.shape)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCALE DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# log1p scale on spectra only (labels)
transformer = FunctionTransformer(np.log1p, validate=True)
transformer.fit_transform(labels_raw)

# minmax scale on both 
scaler = pp.MinMaxScaler()
values = scaler.fit_transform(np.array(values_raw)).astype(np.float32)
labels = scaler.fit_transform(np.array(labels_raw)).astype(np.float32)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SPLIT DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(values, labels, test_size = 0.1)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train_val, y_train_val, test_size = 0.1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RESHAPE AND SAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# create the final format
train_append = np.append(X_train, y_train, axis=1)
val_append = np.append(X_val, y_val, axis=1)
test_append = np.append(X_test, y_test, axis=1)

train_data = train_append.flatten()
val_data = val_append.flatten()
test_data = test_append.flatten()

# save 
now = str(datetime.now())
np.save('train_data' + str(now), train_data)
np.save('val_data' + str(now), val_data)
np.save('test_data' + str(now), test_data)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OPTIONAL CHECKS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

print("X_train only:", X_train.shape)
print("X_val only: ", X_val.shape)
print("X_test only: ", X_test.shape)
print("y_train only: ", y_train.shape)
print("y_val only: ", y_val.shape)
print("y_test only: ", y_test.shape)
print("train append shape", train_append.shape)
print("val append shape", val_append.shape)
print("test append shape", test_append.shape)

print("Numpy files written")