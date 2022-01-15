# source /etc/bashrc

#!/bin/bash -l 
python -m pytest -s --junitxml=results.xml
exit 0
