import pytest
import operator, os
import common_code


def error_if(f, f_symbol, data_field, actual_values, expected_values,
             model_name, errors, all_values, frequency_str):
  d = actual_values[data_field]
  if f_symbol == '<':
    # Every time a value is smaller, update archive_value
    archive_value = float('inf')
  elif f_symbol == '>':
    # Every time a value is greater, update archive_value
    archive_value = float('-inf')
  else:
    raise Exception('Invalid Function Symbol %s' % f_symbol)
  for model_id in sorted(d.keys()):
    for epoch_id in sorted(d[model_id].keys()):
      actual_value = d[model_id][epoch_id]
      expected_value = expected_values[model_name + frequency_str][data_field]

      if actual_value is None:
        errors.append('actual_value: d[%s][%s] is None' % (model_id, epoch_id))
      else:
        print('actual_value={av}'.format(av=actual_value))
      if expected_value is None:
        errors.append(
          'expected_value: d[%s]([%s] is None' % (model_id, epoch_id))
      else:
        print('expected_value={ev}'.format(ev=expected_value))

      if (actual_value is not None) and (expected_value is not None):
        if f(actual_value, expected_value):
          errors.append('%f %s %f %s Model %s Epoch %s %s' % (
            actual_value, f_symbol, expected_value, model_name, model_id,
            epoch_id, data_field))
        all_values.append('%f %s Model %s Epoch %s %s' % (
          actual_value, model_name, model_id, epoch_id, data_field))

        if f(actual_value, archive_value):
          archive_value = actual_value
      else:
        print('archiving: either actual_value or expected_value is None.')
  return archive_value


def run_tests(actual_performance, model_name, dir_name, should_log,
              compiler_name, cluster, frequency_str=''):
  expected_performance = common_code.csv_to_dict(
    '%s/bamboo/integration_tests/expected_values/%s/%s/expected_performance.csv' % (dir_name, cluster, compiler_name))
  errors = []
  all_values = []
  greater_than = operator.gt
  less_than = operator.lt
  max_run_time = error_if(greater_than, '>', 'training_run_time', actual_performance, expected_performance, model_name, errors, all_values, frequency_str)
  max_mean     = error_if(greater_than, '>', 'training_mean', actual_performance, expected_performance, model_name, errors, all_values, frequency_str)
  max_max      = error_if(greater_than, '>', 'training_max', actual_performance, expected_performance, model_name, errors, all_values, frequency_str)
  max_min      = error_if(greater_than, '>', 'training_min', actual_performance, expected_performance, model_name, errors, all_values, frequency_str)
  max_stdev    = error_if(greater_than, '>', 'training_stdev', actual_performance, expected_performance, model_name, errors, all_values, frequency_str)
  min_accuracy = error_if(less_than, '<', 'test_accuracy', actual_performance, expected_performance, model_name, errors, all_values, frequency_str)

  archival_string = '%s, %f, %f, %f, %f, %f, %f\n' % (
    os.environ['bamboo_buildNumber'], max_run_time, max_mean, max_max, max_min,
    max_stdev, min_accuracy)
  print('archival_string: ' + archival_string)
  if os.environ['LOGNAME'] == 'lbannusr':
    key = 'bamboo_planKey'
    if key in os.environ:
      plan = os.environ[key]
      if plan in ['LBANN-NIGHTD', 'LBANN-WD']:
        archive_file = '/usr/workspace/wsb/lbannusr/archives/%s/%s/%s/performance_%s.txt' % (plan, cluster, compiler_name, model_name)
        print('Archive file: ' + archive_file)
        with open(archive_file, 'a') as archive:
          print('Archiving to file.')
          archive.write(archival_string)
      else:
        print('The plan %s does not have archiving activated' % plan)
    else:
      print('%s is not in os.environ' % key)
  else:
    print('os.environ["LOGNAME"]=%s' % os.environ['LOGNAME'])

  print('Errors for: %s %s (%d)' % (model_name, compiler_name, len(errors)))
  for error in errors:
    print(error)
  if should_log:
    print('All values for: %s %s (%d)' % (
      model_name, compiler_name, len(all_values)))
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


def skeleton_performance_lenet_mnist(cluster, dir_name, executables,
                                     compiler_name):
  if compiler_name not in executables:
    e = 'skeleton_performance_lenet_mnist: default_exes[%s] does not exist' % compiler_name
    print('Skip - ' + e)
    pytest.skip(e)
  executable = executables[compiler_name]
  model_name = 'lenet_mnist'
  model_folder = 'models/' + model_name
  should_log = True
  actual_performance = common_code.skeleton(
    cluster, dir_name, executable, model_folder, model_name, DATA_FIELDS,
    should_log, compiler_name=compiler_name)
  run_tests(actual_performance, model_name, dir_name, should_log,
            compiler_name, cluster)


def skeleton_performance_alexnet(cluster, dir_name, executables, compiler_name,
                                 weekly):
  if compiler_name not in executables:
    e = 'skeleton_performance_alexnet: default_exes[%s] does not exist' % compiler_name
    print('Skip - ' + e)
    pytest.skip(e)
  executable = executables[compiler_name]
  model_name = 'alexnet'
  model_folder = 'models/' + model_name
  should_log = True
  actual_performance = common_code.skeleton(
    cluster, dir_name, executable, model_folder, model_name, DATA_FIELDS,
    should_log, compiler_name=compiler_name, weekly=weekly)
  frequency_str = '_nightly'
  if weekly:
    frequency_str = '_weekly'
  run_tests(actual_performance, model_name, dir_name, should_log,
            compiler_name, cluster, frequency_str)


def skeleton_performance_full_alexnet(cluster, dir_name, executables,
                                      compiler_name, weekly, run):
  if not run:
    e = 'skeleton_performance_full_alexnet: Ignored'
    print('Skip - ' + e)
    pytest.skip(e)
  if not weekly:
    e = 'skeleton_performance_full_alexnet: Non-local testing'
    print('Skip - ' + e)
    pytest.skip(e)
  if compiler_name not in executables:
    e = 'skeleton_performance_full_alexnet: default_exes[%s] does not exist' % compiler_name
    print('Skip - ' + e)
    pytest.skip(e)
  executable = executables[compiler_name]
  if not os.path.exists(executable):
    pytest.skip('Executable does not exist: %s' % executable)
  model_name = 'full_alexnet'
  should_log = True
  output_file_name = '%s/bamboo/integration_tests/output/%s_%s_output.txt' %(dir_name, model_name, compiler_name)
  error_file_name = '%s/bamboo/integration_tests/error/%s_%s_error.txt' %(dir_name, model_name, compiler_name) 
  if cluster in ['catalyst']:
    command = 'salloc --nodes 128 %s/bamboo/integration_tests/%s.sh > %s 2> %s' % (dir_name, model_name, output_file_name, error_file_name)
  elif cluster in ['pascal', 'ray']:
    e = 'skeleton_performance_full_alexnet: Pascal, Ray are unsupported for skeleton_performance_full_alexnet'
    print('Skip - ' + e)
    pytest.skip(e)
  else:
    raise Exception('Unsupported Cluster %s' % cluster)
  common_code.run_lbann(command, model_name, output_file_name, error_file_name,
                        should_log)  # Don't need return value
  actual_performance = common_code.extract_data(output_file_name, DATA_FIELDS,
                                                should_log)
  run_tests(actual_performance, model_name, dir_name, should_log, compiler_name,
            cluster)


def test_integration_performance_lenet_mnist_clang6(cluster, dirname, exes):
  skeleton_performance_lenet_mnist(cluster, dirname, exes, 'clang6')


def test_integration_performance_alexnet_clang6(cluster, dirname, exes, weekly):
  skeleton_performance_alexnet(cluster, dirname, exes, 'clang6', weekly)


def test_integration_performance_full_alexnet_clang6(cluster, dirname, exes,
                                                     weekly, run):
  skeleton_performance_full_alexnet(cluster, dirname, exes, 'clang6', weekly,
                                    run)


def test_integration_performance_lenet_mnist_gcc7(cluster, dirname, exes):
  skeleton_performance_lenet_mnist(cluster, dirname, exes, 'gcc7')


def test_integration_performance_alexnet_gcc7(cluster, dirname, exes, weekly):
  skeleton_performance_alexnet(cluster, dirname, exes, 'gcc7', weekly)


def test_integration_performance_full_alexnet_gcc7(cluster, dirname, exes,
                                                   weekly, run):
  skeleton_performance_full_alexnet(cluster, dirname, exes, 'gcc7', weekly, run)


def test_integration_performance_lenet_mnist_intel19(cluster, dirname, exes):
  skeleton_performance_lenet_mnist(cluster, dirname, exes, 'intel19')


def test_integration_performance_alexnet_intel19(cluster, dirname, exes,
                                                 weekly):
  skeleton_performance_alexnet(cluster, dirname, exes, 'intel19', weekly)


def test_integration_performance_full_alexnet_intel19(cluster, dirname, exes,
                                                      weekly, run):
  skeleton_performance_full_alexnet(cluster, dirname, exes, 'intel19', weekly,
                                    run)


# Run with python -m pytest -s test_integration_performance.py -k 'test_integration_performance_lenet_mnist_exe' --exe=<executable>
def test_integration_performance_lenet_mnist_exe(cluster, dirname, exe):
    if exe is None:
      e = 'test_integration_performance_lenet_mnist_exe: Non-local testing'
      print('Skip - ' + e)
      pytest.skip(e)
    exes = {'exe': exe}
    skeleton_performance_lenet_mnist(cluster, dirname, exes, 'exe')


# Run with python -m pytest -s test_integration_performance.py -k 'test_integration_performance_alexnet_exe' --exe=<executable>
def test_integration_performance_alexnet_exe(cluster, dirname, exe):
    if exe is None:
      e = 'stest_integration_performance_alexnet_exe: Non-local testing'
      print('Skip - ' + e)
      pytest.skip(e)
    exes = {'exe': exe}
    skeleton_performance_alexnet(cluster, dirname, exes, 'exe', True)


# Run with python -m pytest -s test_integration_performance.py -k 'test_integration_performance_full_alexnet_exe' --exe=<executable>
def test_integration_performance_full_alexnet_exe(cluster, dirname, exe):
    if exe is None:
      e = 'test_integration_performance_full_alexnet_exe: Non-local testing'
      print('Skip - ' + e)
      pytest.skip(e)
    exes = {'exe': exe}
    skeleton_performance_full_alexnet(cluster, dirname, exes, 'exe', True)
