import pytest
import common_code

def error_if(f, f_symbol, data_field, actual_values, expected_values, model_name, errors, all_values):
  d = actual_values[data_field]
  for model_id in sorted(d.keys()):
    for epoch_id in sorted(d[model_id].keys()):
      actual_value = d[model_id][epoch_id]
      expected_value = expected_values[epoch_id][data_field]

      if actual_value == None:
        errors.append('d[%s][%s] == None' % (model_id, epoch_id))
      if expected_value == None:
        errors.append('d[%s]([%s] == None' % (model_id, epoch_id))

      if f(actual_value, expected_value):
        errors.append('%.2f %s %.2f %s Model %s Epoch %s %s' % (actual_value, f_symbol, expected_value, model_name, model_id, epoch_id, data_field))
      all_values.append('%.2f %s Model %s Epoch %s %s' % (actual_value, model_name, model_id, epoch_id, data_field))

def run_tests(actual_objective_functions, model_name, dir_name, should_log):
    expected_objective_functions = common_code.csv_to_dict('%s/bamboo/integration_tests/expected_values/expected_%s_objective_functions.csv' % (dir_name, model_name))
    errors = []
    all_values = []
    epsilon = 0.01
    outside_epsilon = lambda x,y: abs(x - y) > epsilon
    error_if(outside_epsilon, '!=', 'training_objective_function', actual_objective_functions, expected_objective_functions, model_name, errors, all_values) 

    print('Errors for: %s (%d)' % (model_name, len(errors)))
    for error in errors:
        print(error)
    if should_log:
        print('All values for: %s (%d)' % (model_name, len(all_values)))
    for value in all_values:
      print(value)
    assert errors == []

DATA_FIELDS = [
  'training_objective_function'
]

def test_model_conv_autoencoder_mnist(dirname, exe):
    model_folder = 'autoencoder_mnist'
    model_name = 'conv_autoencoder_mnist'
    actual_objective_functions = common_code.skeleton(dirname, exe, model_folder, model_name, DATA_FIELDS, should_log=True)
    run_tests(actual_objective_functions, model_name, dirname, True)
    

def test_model_conv_autoencoder_imagenet(dirname, exe, weekly):
    model_folder = 'autoencoder_imagenet'
    model_name = 'conv_autoencoder_imagenet'
    if weekly:
        actual_objective_functions = common_code.skeleton(dirname, exe, model_folder, model_name, DATA_FIELDS, should_log=True)
        run_tests(actual_objective_functions, model_name, dirname, True)
    else:
        pytest.skip('Not doing weekly testing')
