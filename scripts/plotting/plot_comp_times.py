import sys
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import load_events

def plot_times(files, names, output_name):
  """Plot comparisons of run times.

  This only plots for model 0.

  """
  results = []
  for f in files:
    results.append(load_events.load_values(
      f,
      event_names=['objective_time', 'objective_evaluation_time',
                   'objective_differentiation_time', 'metric_evaluation_time'],
      layer_event_names=['fp_time', 'bp_time', 'update_time', 'imcomm_time', 'opt_time'],
      model=0))
  fig, ax = plt.subplots(1, 1)
  bar_width = 0.35
  labels = ['{0}'.format(x) for x in results[0]['fp_time'].keys()]
  labels += ['obj', 'metrics']
  starts = np.arange(len(labels))*(bar_width*len(files)+0.3) + 1
  for i in range(len(results)):
    result = results[i]
    l = len(result['fp_time'].keys())
    fp_tot = OrderedDict()
    bp_tot = OrderedDict()
    update_tot = OrderedDict()
    imcomm_tot = OrderedDict()
    opt_tot = OrderedDict()
    for layer in result['fp_time'].keys():
      fp_tot[layer] = np.sum(result['fp_time'][layer])
      bp_tot[layer] = np.sum(result['bp_time'][layer])
      update_tot[layer] = np.sum(result['update_time'][layer])
      if 'imcomm_time' in result and layer in result['imcomm_time']:
        imcomm_tot[layer] = np.sum(result['imcomm_time'][layer])
      else:
        imcomm_tot[layer] = 0
      if 'opt_time' in result and layer in result['opt_time']:
        opt_tot[layer] = np.sum(result['opt_time'][layer])
      else:
        opt_tot[layer] = 0
    obj_val_tot = 0.0
    obj_grad_tot = 0.0
    if 'objective_time' in result:
      obj_val_tot = np.sum(result['objective_time'])
    if 'objective_evaluation_time' in result:
      obj_val_tot = np.sum(result['objective_evaluation_time'])
      obj_grad_tot = np.sum(result['objective_differentiation_time'])
    metric_tot = 0.0
    if 'metric_evaluation_time' in result:
      metric_tot = np.sum(result['metric_evaluation_time'])
    fp_tot = np.array(list(fp_tot.values()) + [obj_val_tot, metric_tot])
    bp_tot = np.array(list(bp_tot.values()) + [obj_grad_tot, 0.0])
    update_tot = np.array(list(update_tot.values()) + [0.0, 0.0])
    imcomm_tot = np.array(list(imcomm_tot.values()) + [0.0, 0.0])
    opt_tot = np.array(list(opt_tot.values()) + [0.0, 0.0])
    ax.bar(starts + i*bar_width, fp_tot, bar_width, color='blue')
    ax.bar(starts + i*bar_width, bp_tot, bar_width, bottom=fp_tot,
           color='green')
    ax.bar(starts + i*bar_width, update_tot, bar_width, bottom=fp_tot+bp_tot,
           color='yellow')
    ax.bar(starts + i*bar_width, opt_tot, bar_width,
           bottom=fp_tot+bp_tot+update_tot, color='magenta')
    rects = ax.bar(starts + i*bar_width, imcomm_tot, bar_width,
                   bottom=fp_tot+bp_tot+update_tot+opt_tot, color='red')
    # Add the name to this bar.
    ax.text(rects[0].get_x() + rects[0].get_width() / 2,
            rects[0].get_y() + rects[0].get_height() + 1,
            names[i],
            ha='center', va='bottom', rotation='vertical', fontsize=4)
  ax.set_ylabel('Time (s)')
  ax.set_xticks(starts + bar_width*(len(results)/2))
  ax.set_xticklabels(labels, rotation='vertical')
  if len(fp_tot) > 35:
    for label in ax.xaxis.get_ticklabels():
      label.set_fontsize(3)
  #for label in ax.xaxis.get_ticklabels()[::2]:
  #  label.set_visible(False)
  ax.set_title('Per-layer runtime breakdown')
  ax.legend(('FP', 'BP', 'Update', 'Opt', 'Imcomm'))
  plt.tight_layout()
  plt.savefig(output_name + '.pdf')

if __name__ == '__main__':
  if len(sys.argv) < 4:
    print('plot_comp_times.py: [events] [names] output')
    sys.exit(1)
  num = (len(sys.argv) - 2) // 2
  plot_times(sys.argv[1:num+1], sys.argv[num+1:2*num+1], sys.argv[-1])
