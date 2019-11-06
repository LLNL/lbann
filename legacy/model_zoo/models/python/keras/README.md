## Keras converter tool
kerbann.py is a tool used to convert models written in keras' python format to LBANN's prototext format. It is still in development, and only supports keras models which have direct analogue in lbann.

### Usage
To use kerbann simply import the kerbann.py file into your keras model, and call the converter function keras_to_lbann with the keras model object you wish to convert. Do this *after* calling compile on the keras model. This keras_to_lbann function will also set model wide parameters for your lbann model, either using defaults, or named variables passed to the function. Currently you can set: name (model type), data_layout, mini batch size, block size, epoch count, number of parallel data readers, and processes per model. The function also requires a num_classes argument currently, as most keras models tested use this parameter. It may be remmoved later. 

This function will out the converted keras model to a prototext of the same name, along with a command showing general lbann usage on a slurm system. An example of complete usage can be found in this directory: mnist_cnn.py (keras model), and mnist_cnn.prototext (resulting LBANN model). 
