import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, re, subprocess


def test_compiler_build_script(cluster, dirname):
    # if cluster not in ['catalyst', 'corona', 'lassen', 'pascal', 'ray']:
    #     e = 'test_compiler_build_script: Unsupported Cluster %s' % cluster
    #     print('Skip - ' + e)
    #     pytest.skip(e)
    output_file_name = '%s/bamboo/compiler_tests/output/build_script_output.txt' % (dirname)
    error_file_name = '%s/bamboo/compiler_tests/error/build_script_error.txt' % (dirname)
    if cluster in ['lassen', 'pascal', 'ray']:
        command = '%s/scripts/build_lbann.sh -d -l bamboo --test -- +cuda +deterministic +half +fft +vision +numpy > %s 2> %s' % (
            dirname, output_file_name, error_file_name)
    elif cluster in ['corona']:
        command = '%s/scripts/build_lbann.sh -d -l bamboo --test -- +rocm +deterministic +half +fft +vision +numpy > %s 2> %s' % (
            dirname, output_file_name, error_file_name)
    elif cluster in ['catalyst']:
        command = '%s/scripts/build_lbann.sh -d -l bamboo --test -- +onednn +deterministic +half +fft +vision +numpy > %s 2> %s' % (
            dirname, output_file_name, error_file_name)
    else:
        e = 'test_compiler_build_script: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)
        
    return_code = os.system(command)
    tools.assert_success(return_code, error_file_name)
