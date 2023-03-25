import os
import os.path
import datetime

def make_experiment_dir(job_name=None):
    if job_name is None:
        job_name = 'lbann_siamese'
    if 'LBANN_EXPERIMENT_DIR' in os.environ:
        experiment_dir = os.environ['LBANN_EXPERIMENT_DIR']
    else:
        experiment_dir = os.path.join(os.getcwd())
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(experiment_dir,
                                  '{}_{}'.format(timestamp, job_name))
    i = 1
    while os.path.lexists(experiment_dir):
        i += 1
        experiment_dir = os.path.join(
            os.path.dirname(experiment_dir),
            '{}_{}_{}'.format(timestamp, job_name, i))
    experiment_dir = os.path.abspath(experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir
