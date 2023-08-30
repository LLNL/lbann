import sys
sys.path.insert(0, '../common_python')
import os
import pytest
import tools


def skeleton_jag_reconstruction_loss(cluster, dir_name,
                                     weekly, data_reader_fraction):
    if cluster == 'corona':
      message = f'{os.path.basename(__file__)} is only broken on corona systems'
      print('Skip - ' + message)
      pytest.skip(message)

    output_file_name = '%s/ci_test/unit_tests/output/jag_reconstruction_loss_output.txt' % (dir_name)
    error_file_name  = '%s/ci_test/unit_tests/error/jag_reconstruction_loss_error.txt' % (dir_name)
    command = tools.get_command(
        cluster=cluster,
        num_nodes=2,
        num_processes=32,
        disable_cuda=1,
        dir_name=dir_name,
        sample_list_train_default='/p/vast1/lbann/datasets/JAG/10MJAG/1M_A/100K4trainers/100Kindex.txt',
        sample_list_test_default='/p/vast1/lbann/datasets/JAG/10MJAG/1M_A/100K16trainers/t1_sample_list.txt',
        data_reader_name='jag',
        data_reader_fraction='prototext',
        metadata='applications/physics/data/jag_100M_metadata.prototext',
        model_folder='tests',
        model_name='jag_single_layer_ae',
        optimizer_name='adam',
        output_file_name=output_file_name,
        error_file_name=error_file_name, weekly=weekly)
    return_code = os.system(command)
    tools.assert_success(return_code, error_file_name)

# Run with python3 -m pytest -s test_unit_ridge_regression.py -k 'test_unit_jag_reconstruction_loss'
def test_unit_jag_reconstruction_loss(cluster, dirname,
                                      weekly, data_reader_fraction):
    skeleton_jag_reconstruction_loss(cluster, dirname,
                                     weekly, data_reader_fraction)
