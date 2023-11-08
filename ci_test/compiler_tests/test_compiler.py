import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, re
import shutil

def test_compiler_build_script(cluster, dirname):
    test_base_dir = os.path.join(dirname, 'ci_test', 'compiler_tests')
    output_file_name = os.path.join(test_base_dir, 'output', 'build_script_output.txt')
    error_file_name = os.path.join(test_base_dir, 'error', 'build_script_error.txt')

    # Get environment variables
    ENV_NAME = os.getenv('SPACK_ENV_NAME')
    CLEAN_BUILD_FLAG = os.getenv('CLEAN_BUILD_FLAG')
    clean_flag = ''
    if CLEAN_BUILD_FLAG is not None:
        clean_flag = '--clean-build'

    SPACK_DEPS_FLAG = os.getenv('SPACK_DEPS_FLAG')
    if SPACK_DEPS_FLAG is None:
        print('Unknown state: MISSING SPACK_DEPS_FLAG')
        SPACK_DEPS_FLAG='-r'

    common_cmd = '%s/scripts/build_lbann.sh --ci %s -l %s -j $(($(nproc)+2)) %s -p py-scipy@1.8.1 --pip pytest --pip tqdm -- +deterministic +vision +numpy +unit_tests' % (dirname, SPACK_DEPS_FLAG, ENV_NAME, clean_flag)
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

    artifact_dir = os.path.join(test_base_dir, 'output')
    with os.scandir(dirname) as it:
        for entry in it:
            if entry.is_file() and re.match(r'spack-.*txt', entry.name):
                (base, ext) = os.path.splitext(entry.name)
                new_file_name = base + '_output' + ext
                shutil.copyfile(entry.path, os.path.join(artifact_dir, new_file_name))

    tools.assert_success(return_code, error_file_name)
