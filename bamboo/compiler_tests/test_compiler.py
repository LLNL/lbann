import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, re

def test_compiler_build_script(cluster, dirname):
    bamboo_base_dir = '%s/bamboo/compiler_tests' % (dirname)
    output_file_name = '%s/output/build_script_output.txt' % (bamboo_base_dir)
    error_file_name = '%s/error/build_script_error.txt' % (bamboo_base_dir)

    # Get environment variables
    BAMBOO_AGENT = os.getenv('bamboo_agentId')

    common_cmd = '%s/scripts/build_lbann.sh -d -l bamboo-%s --test --clean-build -j $(($(nproc)+2)) -- +deterministic +vision +numpy' % (dirname, BAMBOO_AGENT)
    if cluster in ['lassen', 'pascal', 'ray']:
        command = '%s +cuda +half +fft > %s 2> %s' % (common_cmd, output_file_name, error_file_name)
    elif cluster in ['corona']:
        command = '%s +rocm > %s 2> %s' % (common_cmd, output_file_name, error_file_name)
    elif cluster in ['catalyst']:
        command = '%s +onednn +half +fft > %s 2> %s' % (common_cmd, output_file_name, error_file_name)
    else:
        e = 'test_compiler_build_script: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)

    return_code = os.system(command)

    gather_artifacts_cmd = 'cp %s/spack-*.txt %s/output' % (dirname, bamboo_base_dir)
    os.system(gather_artifacts_cmd)

    tools.assert_success(return_code, error_file_name)

