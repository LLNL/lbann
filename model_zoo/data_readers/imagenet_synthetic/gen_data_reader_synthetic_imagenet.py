#!/usr/bin/env python3

import sys

file_name_prefix="data_reader_synthetic_imagenet"
template='''\
data_reader {{
  reader {{
    name: "synthetic"
    role: "train"
    shuffle: true
    num_samples: {NUM_SAMPLES}
    num_labels: 1000
    synth_dimensions: "3 224 224"
    validation_percent: 0
    absolute_sample_count: 0
    percent_of_data_to_use: 1.0
  }}

  reader {{
    name: "synthetic"
    role: "test"
    shuffle: true
    num_samples: 1024
    num_labels: 1000
    synth_dimensions: "3 224 224"
    absolute_sample_count: 0
    percent_of_data_to_use: 1.0
  }}
}}
'''

num_samples = int(sys.argv[1])

fn = file_name_prefix + "_" + str(num_samples) + ".prototext"
print("Generating " + fn)
f = open(fn, 'w')
f.write(template.format(NUM_SAMPLES=str(num_samples)))
f.close()

