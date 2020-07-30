import numpy as np
import sys
import glob

#Check if there are scalars with all zero values
#Input is scalar values dumped from LBANN input layer
fdir = sys.argv[1]
epoch = sys.argv[2]
print(fdir)
scalar_files = glob.glob(fdir+"*training-epoch"+str(epoch)+"*gt_sca*.npy")
scalar_jag = np.load(scalar_files[0])
print("First JAG param shape " , scalar_jag.shape)
print("param jag ", scalar_jag)
for i, f in enumerate(scalar_files):
        if(i > 0) :
           scalar_jag = np.concatenate((scalar_jag, np.load(f)))

print("Final JAG param shape " , scalar_jag.shape)


num_cols = scalar_jag.shape[1]
print("Num cols ", num_cols)

zeros =  np.where(np.all(np.isclose(scalar_jag, 0), axis=1))
print("Num of zerors ", zeros[0].shape , "   ", zeros)
