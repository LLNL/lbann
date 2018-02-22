# Building Autoencoder in LBANN: A CANDLE Example
(by [Sam Ade Jacobs](https://people.llnl.gov/jacobs32))

CANcer Distributed Learning Enviroment ([CANDLE](http://candle.cels.anl.gov/)) is a collaborative project between US DOE national laboratories and National Cancer Institute (NCI) aims at enabling high-performance deep learning in support of the DOE-NCI Cancer project. As a partner, researchers at [LLNL](https://www.llnl.gov) are developing open-source HPC deep learning toolkit called [LBANN](https://github.com/LLNL/lbann) to support **CANDLE** and other projects. 

[Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) is one of deep learning techniques being explored in the **CANDLE** team. In this blog, I will explain how to build autoencoder of interest to **CANDLE** project within LBANN framework. Examples in this blog were taken from Tensorflow version of similar deep learning network architecture provided by the **CANDLE** research team.

## Autoencoder in LBANN
A network architecture in LBANN is a collection of layers as a sequential list or graph. To build an autoencoder model in LBANN, the user simply describe how the layers are connected in a [model prototext file](https://github.com/LLNL/lbann/tree/develop/model_zoo/models/autoencoder_candle_pilot1), provide training optimization paratemers in the [optimizer prototext file](https://github.com/LLNL/lbann/tree/develop/model_zoo/optimizers), and input data (and labels in case of classification) in the [data reader prototext file ](https://github.com/LLNL/lbann/tree/develop/model_zoo/data_readers). The prototext files provide the flexibility for users to change a number of network and optimization hyperparameters at run time. For example, an LBANN fully connected (also known as linear or inner product in other deep learning toolkits) layer can be described as shown:
 ```
layer {
    index: 8
    parent: 7
    data_layout: "data_parallel"
    fully_connected {
      num_neurons: 5000
      weight_initialization: "glorot_uniform"
      has_bias: true
    }
  }
  
  ```

Most of the attributes are self descriptive and some of them can be changed "on-the-fly". For instance, the glorot_uniform weight initialization scheme can be replaced with other schemes such as uniform, normal, he_normal, he_uniform, glorot_normal and so on.


## Execute LBANN Autoencoder Example on LC
LBANN has a number of prototext files to support the CANDLE project, one example is provided here. Users can leverage on existing examples as deemed fit. To execute available examples on Livermore Computing (LC) machines:
   1. First install LBANN (detailed instructions available [here](https://github.com/LLNL/lbann.git))
   2. Allocate compute resources using SLURM: `salloc -N12 -t 60`
   3. Run a CANDLE test experiment from the main lbann directory using the following command:
 ```
  srun -n48 build/gnu.flash.llnl.gov/lbann/build/model_zoo/lbann \
--model=model_zoo/models/autoencoder_candle_pilot1/model_autoencoder_chem_ecfp_500x250x100.prototext \
--reader=model_zoo/data_readers/data_reader_candle_pilot1.prototext \
--optimizer=model_zoo/optimizers/opt_adam.prototext
```
  The 20th epoch training should produce the following results (~90% Pearson correlation coefficient on test dataset) on Flash:
  
```
--------------------------------------------------------------------------------
[19] Epoch : stats formated [tr/v/te] iter/epoch = [1289/144/160]
            global MB = [1024/1024/1024] global last MB = [ 806  / 203  / 113  ]
             local MB = [1024/1024/1024]  local last MB = [ 806+0/ 203+0/ 113+0]
--------------------------------------------------------------------------------
Model 0 training epoch 19 objective function : 0.00852335
Model 0 training epoch 19 Pearson correlation : 0.903967
Model 0 training epoch 19 run time : 76.3258s
Model 0 training epoch 19 mini-batch time statistics : 0.0592126s mean, 0.135559s max, 0.0570875s min, 0.00623434s stdev
Model 0 validation objective function : 0.00865012
Model 0 validation Pearson correlation : 0.902541
Model 0 validation run time : 6.18625s
Model 0 validation mini-batch time statistics : 0.0429592s mean, 0.0501507s max, 0.0151007s min, 0.00245794s stdev
Model 0 test objective function : 0.00862292
Model 0 test Pearson correlation : 0.902876
Model 0 test run time : 6.85016s
Model 0 test mini-batch time statistics : 0.0428128s mean, 0.0476s max, 0.010742s min, 0.0025836s stdev
```
  LBANN performance will vary on a machine to machine basis. Results will also vary, but should not do so significantly. 

## Running on Non-LC Systems
Launch an MPI job using the proper command for your system (srun, mpirun, mpiexec etc), calling the lbann executable found in lbann/build/$YourBuildSys/model_zoo. This executable requires three command line arguments. These arguments are prototext files specifying the model, optimizer and data reader for the execution. Data directories are hardcoded, make sure you have appropriate permission. Models and other hyperparameters can be adjusted by altering appropriate files. 
```
