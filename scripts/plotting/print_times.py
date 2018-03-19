import sys
import numpy as np
import load_events

def print_layer_times(filename, only_model = -1):
  """Print a breakdown of runtimes for each layer in each model."""
  results = load_events.load_values(
    filename,
    event_names=['minibatch_time', 'objective_evaluation_time', 'objective_differentiation_time'],
    layer_event_names=['fp_time', 'bp_time', 'update_time', 'imcomm_time', 'opt_time'],
    model=-1)
  for model in results.keys():
    if model != only_model and only_model != -1:
      continue
    print('Model {0}:'.format(model))
    fp_tot = 0.0
    bp_tot = 0.0
    update_tot = 0.0
    imcomm_tot = 0.0
    opt_tot = 0.0
    for layer in results[model]['fp_time'].keys():
      fp_mean = np.mean(results[model]['fp_time'][layer])
      l_fp_tot = np.sum(results[model]['fp_time'][layer])
      bp_mean = np.mean(results[model]['bp_time'][layer])
      l_bp_tot = np.sum(results[model]['bp_time'][layer])
      update_mean = np.mean(results[model]['update_time'][layer])
      l_update_tot = np.sum(results[model]['update_time'][layer])
      imcomm_mean = 0.0
      l_imcomm_tot = 0.0
      if 'imcomm_time' in results[model] and layer in results[model]['imcomm_time']:
        imcomm_mean = np.mean(results[model]['imcomm_time'][layer])
        l_imcomm_tot = np.sum(results[model]['imcomm_time'][layer])
      opt_mean = 0.0
      l_opt_tot = 0.0
      if 'opt_time' in results[model] and layer in results[model]['opt_time']:
        opt_mean = np.mean(results[model]['opt_time'][layer])
        l_opt_tot = np.sum(results[model]['opt_time'][layer])
      fp_tot += l_fp_tot
      bp_tot += l_bp_tot
      update_tot += l_update_tot
      imcomm_tot += l_imcomm_tot
      opt_tot += l_opt_tot
      portion = imcomm_mean / (fp_mean + bp_mean + update_mean + imcomm_mean + opt_mean) * 100
      print('Layer {0}:\tfp={1:<10.4}\tbp={2:<10.4}\tupdate={3:<10.4}\topt={4:<10.4}\timcomm={5:<10.4}\tportion={6:.4}%'.format(
        layer, fp_mean, bp_mean, update_mean, opt_mean, imcomm_mean, portion))
      print(' '*len('layer {0}'.format(layer)) +
            ':\tfp={0:<10.4}\tbp={1:<10.4}\tupdate={2:<10.4}\topt={3:<10.4}\timcomm={4:<10.4}\tportion={5:.4}%'.format(
              l_fp_tot, l_bp_tot, l_update_tot, l_opt_tot, l_imcomm_tot,
              l_imcomm_tot / (l_fp_tot + l_bp_tot + l_update_tot + l_opt_tot + l_imcomm_tot) * 100))
    print('Total: fp={0:.4} bp={1:.4} update={2:.4} opt={3:.4} imcomm={4:.4} portion={5:.4}%'.format(
      fp_tot, bp_tot, update_tot, opt_tot, imcomm_tot,
      imcomm_tot / (fp_tot + bp_tot + update_tot + opt_tot + imcomm_tot) * 100))
    print('mbavg={0:.4} mbtot={1:.6} objvalavg={2:.4} objvaltot={3:.6} objgradavg={4:.4} objgradtot={5:.6}'.format(
      np.mean(results[model]['minibatch_time']), np.sum(results[model]['minibatch_time']),
      np.mean(results[model]['objective_evaluation_time']),
      np.sum(results[model]['objective_evaluation_time']),
      np.mean(results[model]['objective_differentiation_time']),
      np.sum(results[model]['objective_differentiation_time'])))

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('print_times.py: event file [model]')
    sys.exit(1)
  only_model = -1
  if len(sys.argv) > 2:
    only_model = int(sys.argv[2])
  print_layer_times(sys.argv[1], only_model)
