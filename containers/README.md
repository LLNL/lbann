 ## Singularity
 
 [Singularity](http://singularity.lbl.gov/)
 
 First build a Singularity container with the lbann.def file:
 ```
 sudo singularity build --writable lbann.img lbann.def
 ```
 *Note: Building the image requires root access.*
 
 *Note: --writable allows users to make changes inside the container (Required for LC).*

This will create a container called lbann.img which can be used to invoke lbann on any system with singularity and openmpi installed. 
### Customizing Configuration in lbann.def
Singularity is designed to take advantage of underlying HPC resources. The lbann.def file in this directory specifically installs packages necessary for infiniband interconnects (lines 15-19). It builds openmpi outside of the spack step to ensure it is built with infiniband support (lines 37-55). Experienced users should modify these sections to match with the underlying resources they intend to run on. This defintion file also builds gcc version 4.9.3, and uses it to build openmpi and lbann (lines 33-35). This is also customized to run on specific LC resources, and can be modified depending on the users system. 
 ### Running LBANN with Singualrity
 To run LBANN use mpirun and singularity's execute command:
 ```
 salloc -N2
 mpirun -np 4 singularity exec -B /p:/p lbann.img /lbann/spack_builds/singularity_optimizied_test/model_zoo/lbann  mpirun -np 4 singularity exec -B /p:/p lbann.img /lbann/spack_builds/singularity/model_zoo/lbann  --model=/lbann/model_zoo/tests/model_mnist_distributed_io.prototext --reader=/lbann/model_zoo/data_readers/data_reader_mnist.prototext --optimizer=/lbann/   model_zoo/optimizers/opt_adagrad.prototext
 ```
*Note: The -B singularity command, binds directories from the surrounding filesystem to the container. Be sure to include any necessary files using this command (i.e model prototext files, datasets, etc). Alternatively, system admins are capable of allowing a singularity container to utilize the host's filesystem. This is done by changing MOUNT HOSTFS in the singularity config file.*

 ## Docker
 
  [Docker](https://www.docker.com/)
  
 First build a Docker image with the Dockerfile. From whichever directory contains the Dockerfile:
 ```
docker build -t dockban .
 ```
 
 *Note: The -t flag specifies an identifying tag for this image. "dockban" can be changed to any desired tag.*
 
 ### Customizing Configuration in Dockerfile
 The Dockerfile container defintion is less complicated than its Singularity counterpart. gcc 7.1.0 is built and registered with spack in lines 19-21. Users can change this, as well as LBANN specific build options in spack (line 22). For instance, to add gpu support a user can add "+gpu" to this line. 
 
 ### Running LBANN with Docker
This LBANN build also uses openmpi, so lbann can be launched with mpirun here as well. However, this example will just show the single process invocation. 

Start a docker container from the previously created image, and attach to it. Make sure to bind any necessary directories using -v:
```
docker run -it -v $HOME/MNIST:/MNIST dockban
```
Run LBANN as you would outside of a container:
```
./spack_build/docker_build/model_zoo/lbann --model=model_zoo/models/lenet_mnist/model_lenet_mnist.prototext                  --reader=model_zoo/data_readers/data_reader_mnist.prototext --optimizer=model_zoo/optimizers/opt_sgd.prototext 
```
