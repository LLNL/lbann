#!/bin/sh

# Add Python packages used by the LBANN auxiliary tools
spack add py-configparser ${CENTER_COMPILER}
spack add py-graphviz@0.10.1: ${CENTER_COMPILER}
spack add py-matplotlib@3.0.0: ${CENTER_COMPILER}
spack add py-numpy@1.16.0: ${CENTER_COMPILER}
spack add py-onnx@1.3.0: ${CENTER_COMPILER}
spack add py-pandas@0.24.1: ${CENTER_COMPILER}
spack add py-texttable@1.4.0: ${CENTER_COMPILER}
