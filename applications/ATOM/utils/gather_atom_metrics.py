import sys
import numpy as np
import re

#tag = sys.argv[len(sys.argv)-1]

#for each log file
for num in range(len(sys.argv)-1):
  inp = sys.argv[num+1]
  print("File#", num , " ", inp)
  total_time = 0
  total_train_time = 0
  trainer_metrics = dict()
  results = {}
  partial_results = {}
  current_epoch = {} # Dict for each trainer to track the current epoch
  ds_times = {}
  active_ds_mode = ''
  sync_time = 0
  # Patterns for key metrics
  p_train_time = re.compile('\w+\s+\(instance ([0-9]*)\) training epoch ([0-9]*) run time : ([0-9.]+)')
  p_test_time = re.compile('\w+\s+\(instance ([0-9]*)\) test run time : ([0-9.]+)')
  p_test_recon = re.compile('\w+\s+\(instance ([0-9]*)\) test recon : ([0-9.]+)')
  # Patterns for secondary metrics
  p_train_mb_time = re.compile('\w+\s+\(instance ([0-9]*)\) training epoch ([0-9]*) mini-batch time statistics : ([0-9.]+)s mean')
  # p_train_recon = re.compile('\w+\s+\(instance ([0-9]*)\) training epoch ([0-9]*) recon : ([0-9.]+)')
  # Capture the time required to load the data
  p_preload_data_store_mode = re.compile('starting do_preload_data_store.*num indices:\s+([0-9,]+) for role: (\w+)')
  p_preload_data_store_time = re.compile('\s+do_preload_data_store time:\s+([0-9.]+)')
  # Find the line with time to synchronize trainers
  p_sync_time = re.compile('synchronizing trainers... ([0-9.]+)s')
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
          if not 'test_recon' in trainer_metrics[e][tid].keys():
            trainer_metrics[e][tid]['test_recon'] = r
          else:
            print('@epoch ' + e
                  + ' - duplicate test reconstruction metric found - existing = '
                  +  trainer_metrics[e][tid]['test_recon']
                  + ' discarding ' + r + ' (ran test twice???)')


      m_test_time = p_test_time.match(line)
      if (m_test_time):
          tid = m_test_time.group(1)
          e = current_epoch[tid]
          r = m_test_time.group(2)
          if not 'test_time' in trainer_metrics[e][tid].keys():
            trainer_metrics[e][tid]['test_time'] = r
          else:
            print('@epoch ' + e
                  + ' - duplicate test time found                  - existing = '
                  +  trainer_metrics[e][tid]['test_time']
                  + ' discarding ' + r + ' (ran test twice???)')

      m_train_mb_time = p_train_mb_time.match(line)
      if (m_train_mb_time):
          tid = m_train_mb_time.group(1)
          e = current_epoch[tid]
          if not e == m_train_mb_time.group(2):
            assert('Epoch mismatch')
          r = m_train_mb_time.group(3)
          if not 'train_mb_time' in trainer_metrics[e][tid].keys():
            trainer_metrics[e][tid]['train_mb_time'] = r
          else:
            print('@epoch ' + e
                  + ' - duplicate train mb time found - existing = '
                  +  trainer_metrics[e][tid]['train_mb_time']
                  + ' discarding ' + r + ' (abort)')
            exit(-1)

      m_ds_mode = p_preload_data_store_mode.match(line)
      if (m_ds_mode):
        active_mode = m_ds_mode.group(2)
        samples = int(m_ds_mode.group(1).replace(',', ''))
        ds_times[active_mode] = {'samples' : samples }

      m_ds_time = p_preload_data_store_time.match(line)
      if (m_ds_time):
        time = float(m_ds_time.group(1))
        ds_times[active_mode]['load_time'] = time

      m_sync_time = p_sync_time.match(line)
      if (m_sync_time):
        sync_time = float(m_sync_time.group(1))

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
  # For each epoch, gather data from all trainers
  for e, data in trainer_metrics.items():
    train_times = []
    test_recons = []
    test_times = []
    train_mb_times = []
    # train_recon = []
    num_trainers = len(data)
    # For each trainer in the epoch
    for k, v in data.items():
      if v.keys() == { 'train_time', 'test_recon', 'test_time', 'train_mb_time' }:
        train_times.append(float(v['train_time']))
        test_recons.append(float(v['test_recon']))
        test_times.append(float(v['test_time']))
        train_mb_times.append(float(v['train_mb_time']))

    if num_trainers != len(train_times):
      partial_mean_epoch_train_time = np.mean(np.array(train_times))
      partial_mean_epoch_test_time = np.mean(np.array(test_times))
      partial_total_time = (mean_epoch_train_time + mean_epoch_test_time)
      partial_results[e] = { 'epoch' : e,
                             'total_time' : total_time + partial_total_time,
                             'total_train_time' : total_train_time,
                             'mean_train_time' : partial_mean_epoch_train_time,
                             'std_train_time' : np.std(np.array(train_times)),
                             'min_train_time' : np.amin(np.array(train_times)),
                             'max_train_time' : np.amax(np.array(train_times)),
                             'recon_min' : np.amin(np.array(test_recons)),
                             'recon_max' : np.amax(np.array(test_recons)),
                             'recon_mean' : np.mean(np.array(test_recons)),
                             'recon_std' : np.std(np.array(test_recons)),
                             'mean_test_time' : partial_mean_epoch_test_time,
                             'std_test_time' : np.std(np.array(test_times)),
                             'mean_train_mb_time' : np.mean(np.array(train_mb_times)),
                             'std_train_mb_time' : np.std(np.array(train_mb_times)),
                             'num_trainers': len(train_times)}
                     # 'train_recon_min' : np.amin(np.array(train_recons)),
                     # 'train_recon_max' : np.amax(np.array(train_recons)),
                     # 'train_recon_mean' : np.mean(np.array(train_recons)),
                     # 'train_recon_std' : np.std(np.array(train_recons))
      continue
    else:
      total_train_times.append(train_times)
      if e != '0':
        total_train_times_not_first_epoch.append(train_times)
      mean_epoch_train_time = np.mean(np.array(train_times))
      mean_epoch_test_time = np.mean(np.array(test_times))
      total_time += (mean_epoch_train_time + mean_epoch_test_time)
      total_train_time += mean_epoch_train_time
      results[e] = { 'epoch' : e,
                     'total_time' : total_time,
                     'total_train_time' : total_train_time,
                     'mean_train_time' : mean_epoch_train_time,
                     'std_train_time' : np.std(np.array(train_times)),
                     'min_train_time' : np.amin(np.array(train_times)),
                     'max_train_time' : np.amax(np.array(train_times)),
                     'recon_min' : np.amin(np.array(test_recons)),
                     'recon_max' : np.amax(np.array(test_recons)),
                     'recon_mean' : np.mean(np.array(test_recons)),
                     'recon_std' : np.std(np.array(test_recons)),
                     'mean_test_time' : mean_epoch_test_time,
                     'std_test_time' : np.std(np.array(test_times)),
                     'mean_train_mb_time' : np.mean(np.array(train_mb_times)),
                     'std_train_mb_time' : np.std(np.array(train_mb_times)),
                     'num_trainers': len(train_times)}
                     # 'train_recon_min' : np.amin(np.array(train_recons)),
                     # 'train_recon_max' : np.amax(np.array(train_recons)),
                     # 'train_recon_mean' : np.mean(np.array(train_recons)),
                     # 'train_recon_std' : np.std(np.array(train_recons))

  for e in sorted(results.keys()):
    r = results[e]
    print('Epoch ' + r['epoch']
          + ' {:7.1f}s'.format(r['total_time'])
          + ' / {:7.1f}s'.format(r['total_train_time'])
          + ' training = {:6.2f}s +- {:3.2f} / min = {:6.3f} / max = {:6.3f}'.format(
            r['mean_train_time'], r['std_train_time'], r['min_train_time'], r['max_train_time'])
          + ' :: reconstruction min = {:6.3f} / max = {:6.3f} / avg = {:6.3f} +- {:3.2f}'.format(
            r['recon_min'], r['recon_max'], r['recon_mean'], r['recon_std'])
          + ' :: test time = {:6.3f}s +- {:3.2f}'.format(r['mean_test_time'], r['std_test_time'])
          + ' :: train MB time = {:5.3f}s +- {:3.2f}'.format(r['mean_train_mb_time'], r['std_train_mb_time'])
          + ' :: ' + str(r['num_trainers']) + ' trainers')
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

  print('--------------------------------------------------------------------------------')
  print('Time to load data:')
  for k,v in ds_times.items():
    print('Loading {:12s}'.format(k) + ' data set with {:9d} samples'.format(v['samples']) + ' took {:6.2f}s'.format(v['load_time']))
  print('Time to synchronize the trainers: {:12.6f}s'.format(sync_time))

  for e in sorted(partial_results.keys()):
    r = partial_results[e]
    print('--------------------------------------------------------------------------------')
    print('Results for epochs with only some trainers reporting')
    print('Epoch ' + r['epoch']
          + ' {:7.1f}s'.format(r['total_time'])
          + ' training = {:6.2f}s +- {:3.2f} / min = {:6.3f} / max = {:6.3f}'.format(
            r['mean_train_time'], r['std_train_time'], r['min_train_time'], r['max_train_time'])
          + ' :: reconstruction min = {:6.3f} / max = {:6.3f} / avg = {:6.3f} +- {:3.2f}'.format(
            r['recon_min'], r['recon_max'], r['recon_mean'], r['recon_std'])
          + ' :: test time = {:6.3f}s +- {:3.2f}'.format(r['mean_test_time'], r['std_test_time'])
          + ' :: train MB time = {:5.3f}s +- {:3.2f}'.format(r['mean_train_mb_time'], r['std_train_mb_time'])
          + ' :: ' + str(r['num_trainers']) + ' trainers')
          # + ' :: train reconstruction min = {:6.3f} / max = {:6.3f} / avg = {:6.3f} +- {:3.2f}'.
          # format(r['train_recon_min'], r['train_recon_max'], r['train_recon_mean'], r['train_recon_std']))

  ifile1.close()


#table = pd.DataFrame(results)
#table = pd.DataFrame(all_metrics)
#met_file = "gb_metrics" +str(datetime.date.today())+'.csv'
#print("Saving computed metrics to ", met_file)
#table.to_csv(met_file, index=False)
