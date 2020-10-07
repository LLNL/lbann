import sys
import numpy as np
import re

#tag = sys.argv[len(sys.argv)-1]

#for each log file
for num in range(len(sys.argv)-1):
  inp = sys.argv[num+1]
  print(" File# ", num , " ", inp)
  total_time = 0
  trainer_metrics = dict()
  results = {}
  current_epoch = {} # Dict for each trainer to track the current epoch
  # Patterns for key metrics
  p_train_time = re.compile('\w+\s+\(instance ([0-9]*)\) training epoch ([0-9]*) run time : ([0-9.]+)')
  p_test_time = re.compile('\w+\s+\(instance ([0-9]*)\) test run time : ([0-9.]+)')
  p_test_recon = re.compile('\w+\s+\(instance ([0-9]*)\) test recon : ([0-9.]+)')
  # Patterns for secondary metrics
  p_train_mb_time = re.compile('\w+\s+\(instance ([0-9]*)\) training epoch ([0-9]*) mini-batch time statistics : ([0-9.]+)s mean')
  # p_train_recon = re.compile('\w+\s+\(instance ([0-9]*)\) training epoch ([0-9]*) recon : ([0-9.]+)')
  with open(inp) as ifile1:
    for line in ifile1:
      m_time = p_train_time.match(line)
      if (m_time):
          tid = m_time.group(1)
          e = m_time.group(2)
          current_epoch[tid] = e # track the current epoch for each trainer
          t = m_time.group(3)
          if not trainer_metrics :
            trainer_metrics = { e : { tid : { 'train_time' : t } } }
          else:
            if e in trainer_metrics :
              if tid in trainer_metrics[e]:
                trainer_metrics[e][tid]['train_time'] = t
              else:
                trainer_metrics[e][tid] = { 'train_time' : t }
            else:
              trainer_metrics[e] = { tid : { 'train_time' : t } }

      m_test_recon = p_test_recon.match(line)
      if (m_test_recon):
          tid = m_test_recon.group(1)
          e = current_epoch[tid]
          r = m_test_recon.group(2)
          trainer_metrics[e][tid]['test_recon'] = r

      m_test_time = p_test_time.match(line)
      if (m_test_time):
          tid = m_test_time.group(1)
          e = current_epoch[tid]
          r = m_test_time.group(2)
          trainer_metrics[e][tid]['test_time'] = r

      m_train_mb_time = p_train_mb_time.match(line)
      if (m_train_mb_time):
          tid = m_train_mb_time.group(1)
          e = current_epoch[tid]
          if not e == m_train_mb_time.group(2):
            assert('Epoch mismatch')
          r = m_train_mb_time.group(3)
          trainer_metrics[e][tid]['train_mb_time'] = r

      # m_train_recon = p_train_recon.match(line)
      # if (m_train_recon):
      #     tid = m_train_recon.group(1)
      #     e = current_epoch[tid]
      #     if not e == m_train_recon.group(2):
      #       assert('Epoch mismatch')
      #     r = m_train_recon.group(3)
      #     trainer_metrics[e][tid]['train_recon'] = r

  total_train_times = []
  total_train_times_not_first_epoch = []
  for e, data in trainer_metrics.items():
    train_times = []
    test_recons = []
    test_times = []
    train_mb_times = []
    # train_recon = []
    for k, v in data.items():
      train_times.append(float(v['train_time']))
      test_recons.append(float(v['test_recon']))
      test_times.append(float(v['test_time']))
      train_mb_times.append(float(v['train_mb_time']))
      # train_recons.append(float(v['train_recon']))

    total_train_times.append(train_times)
    if e != '0':
      total_train_times_not_first_epoch.append(train_times)
    mean_epoch_train_time = np.mean(np.array(train_times))
    mean_epoch_test_time = np.mean(np.array(test_times))
    total_time += (mean_epoch_train_time + mean_epoch_test_time)
    results[e] = { 'epoch' : e,
                   'total_time' : total_time,
                   'mean_train_time' : mean_epoch_train_time,
                   'std_train_time' : np.std(np.array(train_times)),
                   'recon_min' : np.amin(np.array(test_recons)),
                   'recon_max' : np.amax(np.array(test_recons)),
                   'recon_mean' : np.mean(np.array(test_recons)),
                   'recon_std' : np.std(np.array(test_recons)),
                   'mean_test_time' : mean_epoch_test_time,
                   'std_test_time' : np.std(np.array(test_times)),
                   'mean_train_mb_time' : np.mean(np.array(train_mb_times)),
                   'std_train_mb_time' : np.std(np.array(train_mb_times))}
                   # 'train_recon_min' : np.amin(np.array(train_recons)),
                   # 'train_recon_max' : np.amax(np.array(train_recons)),
                   # 'train_recon_mean' : np.mean(np.array(train_recons)),
                   # 'train_recon_std' : np.std(np.array(train_recons))

  for e in sorted(results.keys()):
    r = results[e]
    print('Epoch ' + r['epoch']
          + ' {:7.1f}s'.format(r['total_time'])
          + ' training = {:6.2f}s +- {:3.2f}'.format(r['mean_train_time'], r['std_train_time'])
          + ' :: reconstruction min = {:6.3f} / max = {:6.3f} / avg = {:6.3f} +- {:3.2f}'.
          format(r['recon_min'], r['recon_max'], r['recon_mean'], r['recon_std'])
          + ' :: test time = {:6.3f}s +- {:3.2f}'.format(r['mean_test_time'], r['std_test_time'])
          + ' :: train MB time = {:5.3f}s +- {:3.2f}'.format(r['mean_train_mb_time'], r['std_train_mb_time']))
          # + ' :: train reconstruction min = {:6.3f} / max = {:6.3f} / avg = {:6.3f} +- {:3.2f}'.
          # format(r['train_recon_min'], r['train_recon_max'], r['train_recon_mean'], r['train_recon_std']))

  print("All epochs (including 0) epoch time : mean="
        + '{:7.2f}s'.format(np.mean(np.array(total_train_times)))
        + ' +- {:3.2f}'.format(np.std(np.array(total_train_times)))
        + ' min={:6.2f}s'.format(np.amin(np.array(total_train_times)))
        + ' max={:6.2f}s'.format(np.amax(np.array(total_train_times))))
  print("All epochs    (except 0) epoch time : mean="
        + '{:7.2f}s'.format(np.mean(np.array(total_train_times_not_first_epoch)))
        + ' +- {:3.2f}'.format(np.std(np.array(total_train_times_not_first_epoch)))
        + ' min={:6.2f}s'.format(np.amin(np.array(total_train_times_not_first_epoch)))
        + ' max={:6.2f}s'.format(np.amax(np.array(total_train_times_not_first_epoch))))

  ifile1.close()


#table = pd.DataFrame(results)
#table = pd.DataFrame(all_metrics)
#met_file = "gb_metrics" +str(datetime.date.today())+'.csv'
#print("Saving computed metrics to ", met_file)
#table.to_csv(met_file, index=False)
