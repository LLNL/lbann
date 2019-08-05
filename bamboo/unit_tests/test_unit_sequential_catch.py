import os
import pytest

def test_unit_sequential_catch(dirname):
    pytest.skip(e)
    output_file_name = '{d}/bamboo/unit_tests/output/sequential_catch_output.txt'.format(
        d=dirname)
    error_file_name = '{d}/bamboo/unit_tests/error/sequential_catch_error.txt'.format(
        d=dirname)
    command = 'cd ../../build && ctest --no-compress-output -T Test > {o} 2> {e} && cd ../bamboo/unit_tests'.format(
        o=output_file_name, e=error_file_name
    )
    return_code = os.system(command)
    assert return_code == 0
