import os, pytest
import common_code

def error_if(f, f_symbol, data_field, actual_values, expected_values, model_name, errors, all_values):
  d = actual_values[data_field]
  for model_id in sorted(d.keys()):
    for epoch_id in sorted(d[model_id].keys()):
      actual_value = d[model_id][epoch_id]
      expected_value = expected_values[model_name][data_field]

      if actual_value == None:
        errors.append('d[%s][%s] == None' % (model_id, epoch_id))
      if expected_value == None:
        errors.append('d[%s]([%s] == None' % (model_id, epoch_id))

      if f(actual_value, expected_value):
        errors.append('%.2f %s %.2f %s Model %s Epoch %s %s' % (actual_value, f_symbol, expected_value, model_name, model_id, epoch_id, data_field))
      all_values.append('%.2f %s Model %s Epoch %s %s' % (actual_value, model_name, model_id, epoch_id, data_field))

def run_tests(actual_performance, model_name, dir_name, should_log):
  expected_performance = common_code.csv_to_dict('%s/bamboo/integration_tests/expected_values/expected_performance.csv' % dir_name)
  errors = []
  all_values = []
  greater_than = lambda x,y: x > y
  less_than = lambda x,y: x < y
  error_if(greater_than, '>', 'training_run_time', actual_performance, expected_performance, model_name, errors, all_values)
  error_if(greater_than, '>', 'training_mean', actual_performance, expected_performance, model_name, errors, all_values)
  error_if(greater_than, '>', 'training_max', actual_performance, expected_performance, model_name, errors, all_values)
  error_if(greater_than, '>', 'training_min', actual_performance, expected_performance, model_name, errors, all_values)
  error_if(greater_than, '>', 'training_stdev', actual_performance, expected_performance, model_name, errors, all_values)
  error_if(less_than, '<', 'test_accuracy', actual_performance, expected_performance, model_name, errors, all_values)

  print('Errors for: %s (%d)' % (model_name, len(errors)))
  for error in errors:
    print(error)
  if should_log:
    print('All values for: %s (%d)' % (model_name, len(all_values)))
    for value in all_values:
      print(value)
  assert errors == []

DATA_FIELDS = [
  'training_run_time',
  'training_mean',
  'training_max',
  'training_min',
  'training_stdev',
  'test_accuracy'
]
  
def test_performance_lenet_mnist(dirname, exe):
  model_name = 'lenet_mnist'
  actual_performance = common_code.skeleton(dirname, exe, model_name, model_name, DATA_FIELDS, True)
  run_tests(actual_performance, model_name, dirname, True)

def test_performance_alexnet(dirname, exe, weekly):
  if weekly:
    model_name = 'alexnet'
    actual_performance = common_code.skeleton(dirname, exe, model_name, model_name, DATA_FIELDS, True)
    run_tests(actual_performance, model_name, dirname, True)
  else:
    pytest.skip('Not doing weekly testing')

def test_performance_cache_alexnet(dirname, weekly):
  if weekly:
    model_name = 'cache_alexnet'
    output_file_name = '%s_output.txt' % model_name
    command = 'salloc %s/bamboo/integration_tests/%s.sh > %s' % (dirname, model_name, output_file_name)
    common_code.run_lbann(command, model_name, output_file_name, True)
    actual_performance = common_code.extract_data(output_file_name, DATA_FIELDS, True)
    run_tests(actual_performance, model_name, dirname, True)
  else:
    pytest.skip('Not doing weekly testing')
