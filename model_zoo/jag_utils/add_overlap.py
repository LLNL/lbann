#!/usr/tce/bin/python

import sys
import random

if len(sys.argv) < 2 :
  usage = '''
  usage: add_overlap.py list_base_name number_of_lists overlap_percent 


  example: if your lists are t0_list.txt, t1_list.txt and t2_list.txt
           you want 30 percent overlap you would run as:
             add_overlap.py list.txt 2 30
           The output lists names in this example would be:
             t0_list.txt.overlap=30 (etc)
           The output list will contain 30% more samples;
           specifically, t0 will receive 15% of randomly selected
           samples from t1 and t2.
           The input lists are unchanged

           The "excluded" counts in the output files are all set to -1,
           because I haven't taken the time to get them correct.
           I don't think these are used anyplace in lbann, so this should
           be OK.
  '''
  print usage
  exit(9)


#============================================================================
# the List class parses and encapsulate a sample list

class List :

  # the constructor parses the sample list
  def __init__(self, filename) :
    self.filename = filename
    a = open(filename)
    self.first_line = a.readline()
    assert(self.first_line.find('CONDUIT_HDF5_INCLUSION') != -1)
    t = a.readline().split()
    self.valid_samples = int(t[0])
    self.invalid_samples = int(t[1])
    self.num_files = int(t[2])
    self.base_dir = a.readline()
    self.samples = []
    self.counts =  {}
    for line in a :
      if len(line) > 2 :
        t = line.split()
        dir = t[0]
        included = int(t[1])
        excluded = int(t[2])
        self.counts[dir] = included + excluded
        for j in range(3, len(t)):
          self.samples.append((dir, t[j])) 

  #returns a list that contains random samples
  def get_random_samples(self, n) :
    w = set()
    while len(w) < n :
      x = random.randint(0, len(self.samples)-1)
      if x not in w :
        w.add(x)
    r = []
    for x in w :
      r.append(self.samples[x])
    return r  

  def num_samples(self) :
    return len(self.samples)

  # add random samples from some other List to this List
  def add_samples(self, samples) :
     for x in samples :
       self.samples.append(x)

  # write final output (sample list file)
  def write(self, overlap) :
    out = open(self.filename + '.overlap=' + str(overlap), 'w')
    out.write(self.first_line)

    #build map: filename -> (included samples)
    s = {}
    for sample in self.samples :
      if sample[0] not in s :
        s[sample[0]] = set()
      s[sample[0]].add(sample[1])  

    #write included_samples excluded_samples, num_files
    out.write(str(len(self.samples)) + ' -1 ' + str(len(s)) + '\n')
    out.write(self.base_dir)

    #write the samples
    for fn in s.keys() :
      out.write(fn + ' ' + str(len(s[fn])) + ' -1 ')
      for sample_id in s[fn] :
        out.write(sample_id + ' ')
      out.write('\n')
    out.close()

#============================================================================

# parse cmd line
base = sys.argv[1]
count = int(sys.argv[2])
overlap = int(sys.argv[3])

the_lists = []
random_samples = []
for j in range(count) :
  # instantiate a List object; this holds all information from a sample list
  c = List('t' + str(j) + '_' + base)
  the_lists.append(c)

  # get the random samples from the list; this is the overlap that
  # will be added to the other lists
  n = c.num_samples()
  p = int( (overlap / (count-1))* n / 100)
  random_samples.append(c.get_random_samples(p))

# add overlap to the samples
for j in range(count) :
  for k in range(count) :
    if j != k :
      the_lists[j].add_samples(random_samples[k])

# write output files
for x in the_lists :
  x.write(overlap)
