import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, re, subprocess


def test_compiler_build_script(cluster, dirname):
    if cluster not in ['catalyst', 'corona', 'lassen', 'pascal', 'ray']:
        e = 'test_compiler_build_script: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)
    output_file_name = '%s/bamboo/compiler_tests/output/build_script_output.txt' % (dirname)
    error_file_name = '%s/bamboo/compiler_tests/error/build_script_error.txt' % (dirname)
    command = '%s/bamboo/compiler_tests/build_script.sh > %s 2> %s' % (
        dirname, output_file_name, error_file_name)
    return_code = os.system(command)
    tools.assert_success(return_code, error_file_name)
