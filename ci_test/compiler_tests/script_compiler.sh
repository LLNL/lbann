# source /etc/bashrc

#!/bin/bash -l
module load cmake/3.9.2
salloc -N1 -t 600 python -m pytest -s --junitxml=results.xml
#salloc -N1 -t 600 python -m pytest -s test_compiler.py -k 'test_compiler_build_script' --junitxml=results.xml
exit 0
