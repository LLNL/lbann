### Directory Structure:

```
model_zoo   
└─── models
│   └─── alexnet
│   └─── mnist
│   └─── resnet50
│   └─── etc
└─── data_readers      
└─── optimizers      
└─── historical      
└─── tests  
```

Users should primarily be interested in the __models__ directory.
Each __models__ subdirectory contains an LBANN prototext file,
sample output, shell and/or python run scripts, etc.

The lbann.cpp file in this directory is the driver that loads and
runs LBANN from prototext files. 

Various model features (fields in the prototext files) can be over-ridden 
at run time, e.g:

    $ runme.py --num_epochs=10 --mini_batch_size=128

To get a listing of fields that can be over-ridden,
and options that determine processor counts, etc, run:
  $ runme.py --help
in any __models__ subdirectory (note: output is extensive,
so pipe it to less or more)
  
The __data_readers__ and __optimizers__ directories contain prototext
files that are used by the run scripts in the model subdirectories.

The __historical__ directories contain models that are pure c++,
i.e, they do not rely on prototext. These models work (as of this
writing), but are being phased out in favor of the prototext paradigm.

The __tests__ directory contains, um, various tests, and are probably
not of immediate concern to end users
