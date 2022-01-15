# source /etc/bashrc

#!/bin/bash -l
python -m pytest -s --junitxml=results.xml
#python -m pytest -s test_unit_check_proto_models.py -k 'test_unit_models_gcc4' --junitxml=results.xml
exit 0
