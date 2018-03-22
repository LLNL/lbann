import pytest
import common_code

def error_if(f, f_symbol, data_field, actual_values, expected_values, model_name, errors, all_values, weekly):
  d = actual_values[data_field]
  frequency_str = '_nightly'
  if weekly:
    frequency_str = '_weekly'
  for model_id in sorted(d.keys()):
    for epoch_id in sorted(d[model_id].keys()):
      actual_value = d[model_id][epoch_id]
      expected_value = expected_values[epoch_id][data_field + frequency_str]

      if actual_value == None:
        errors.append('d[%s][%s] == None' % (model_id, epoch_id))
      if expected_value == None:
        errors.append('d[%s]([%s] == None' % (model_id, epoch_id))

      if f(actual_value, expected_value):
        errors.append('%f %s %f %s Model %s Epoch %s %s' % (actual_value, f_symbol, expected_value, model_name, model_id, epoch_id, data_field))
      all_values.append('%f %s Model %s Epoch %s %s' % (actual_value, model_name, model_id, epoch_id, data_field))

def run_tests(actual_objective_functions, model_name, dir_name, cluster, should_log, weekly):
    expected_objective_functions = common_code.csv_to_dict('%s/bamboo/integration_tests/expected_values/expected_%s_objective_functions_%s.csv' % (dir_name, model_name, cluster))
    errors = []
    all_values = []
    tolerance = 0.05
    # Are we within tolerance * expected_value?
    outside_tolerance = lambda x,y: abs(x - y) > abs(tolerance * y)
    error_if(outside_tolerance, '!=', 'training_objective_function', actual_objective_functions, expected_objective_functions, model_name, errors, all_values, weekly) 

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

def test_integration_model_conv_autoencoder_mnist(cluster, dirname, exes):
    model_folder = 'models/autoencoder_mnist'
    model_name = 'conv_autoencoder_mnist'
    should_log=False
    actual_objective_functions = common_code.skeleton(cluster, dirname, exes['default'], model_folder, model_name, DATA_FIELDS, should_log)
    run_tests(actual_objective_functions, model_name, dirname, cluster, should_log, False)
    

def test_integration_model_conv_autoencoder_imagenet(cluster, dirname, exes, weekly):
    model_folder = 'models/autoencoder_imagenet'
    model_name = 'conv_autoencoder_imagenet'
    should_log = False
    actual_objective_functions = common_code.skeleton(cluster, dirname, exes['default'], model_folder, model_name, DATA_FIELDS, should_log, weekly=weekly)
    run_tests(actual_objective_functions, model_name, dirname, cluster, should_log, weekly)
