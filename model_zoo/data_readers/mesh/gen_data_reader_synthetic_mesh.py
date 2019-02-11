#!/usr/bin/env python3

import sys

file_name_prefix="data_reader_synthetic_mesh"
template='''\
data_reader {{
  reader {{
    name: "synthetic"
    role: "train"
    shuffle: true
    # This is arbitrary.
    num_samples: {NUM_SAMPLES}
    synth_dimensions: "18 {SPATIAL_DIM} {SPATIAL_DIM}"
    synth_response_dimensions: "1 {SPATIAL_DIM} {SPATIAL_DIM}"
    validation_percent: 0
    absolute_sample_count: 0
    percent_of_data_to_use: 1.0
    disable_responses: false
  }}

  reader {{
    name: "synthetic"
    role: "test"
    shuffle: true
    num_samples: 100
    synth_dimensions: "18 {SPATIAL_DIM} {SPATIAL_DIM}"
    synth_response_dimensions: "1 {SPATIAL_DIM} {SPATIAL_DIM}"
    absolute_sample_count: 0
    percent_of_data_to_use: 1.0
    disable_responses: false
  }}
}}
'''

spatial_dim = int(sys.argv[1])
num_samples = int(sys.argv[2])
fn = file_name_prefix + "_" + str(spatial_dim) + "_" + str(num_samples) + ".prototext"
print("Generating " + fn)
f = open(fn, 'w')
f.write(template.format(NUM_SAMPLES=str(num_samples), SPATIAL_DIM=str(spatial_dim)))
f.close()
