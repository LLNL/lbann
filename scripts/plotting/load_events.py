"""
Load Tensorboard events into Python.
Note: This ignores the CRC32 checks.

"""

from collections import defaultdict
import sys
import os.path
import struct
import event_pb2
import summary_pb2

# Define the structs for the header/footer.
header_struct = struct.Struct('QI')
footer_struct = struct.Struct('I')
# Max size to read into memory.
max_read_size = 5*2**30

def _yield_events_read(filename):
  """Yield events from filename, reading one at a time."""
  with open(filename, 'rb') as f:
    while True:
      data = f.read(header_struct.size)
      if len(data) == 0:
        return
      event_len, lencrc = header_struct.unpack(data)
      event_data = f.read(event_len)
      # Ignore the CRC.
      f.read(footer_struct.size)
      event = event_pb2.Event()
      event.ParseFromString(event_data)
      yield event

def _yield_events_mem(filename):
  """Yield events from filename, reading the whole thing into memory first."""
  data = None
  with open(filename, 'rb') as f:
    data = f.read()
  l = len(data)
  if l == 0:
    return
  pos = 0
  while pos < l:
    event_len, lencrc = header_struct.unpack_from(data, pos)
    event = event_pb2.Event.FromString(
      data[pos+header_struct.size:pos+header_struct.size+event_len])
    pos += header_struct.size + event_len + footer_struct.size
    yield event

def yield_events(filename):
  """Yield events from filename."""
  # If we can fit it in memory, prefer to read the entire file first.
  if os.path.getsize(filename) < max_read_size:
    return _yield_events_mem(filename)
  else:
    return _yield_events_read(filename)

def load_values(filename, event_names=None, layer_event_names=None, model=0):
  """Load specified events from filename.

  model specifies the model to load events from; -1 means any model.
  event_names is a sequence of event names not associated with any layer to
  load.
  layer_event_names is a sequence of layer event names to load.

  This assumes events are named in a relatively standard way:
  - event names are of the form modelN/event-name
  - layer event names are of the form modelN/layer-name/event-name
  Note event-name may have additional slashes in it.

  Returns a dictionary of the results.

  """
  if event_names is None:
    event_names = []
  if layer_event_names is None:
    layer_event_names = []
  results = None
  if model < 0:
    results = defaultdict(dict)
  else:
    results = {}
  for e in yield_events(filename):
    if len(e.summary.value) == 0:
      continue  # Skip non-value entries.
    tag = e.summary.value[0].tag
    val = e.summary.value[0].simple_value
    split_tag1 = tag.split('/', 1)  # For general events.
    if len(split_tag1) <= 1:
      continue
    split_tag2 = tag.split('/', 2)  # For layer events.
    # Determine whether this is the appropriate model.
    # len('model') = 5
    model_num = int(split_tag1[0][5:])
    if model >= 0 and model != model_num:
      continue
    tag = tag[tag.find('/')+1:]
    # Determine where in the results dictionary to add.
    base = results
    if model < 0:
      base = results[model_num]
    event_name = split_tag1[1]
    # Check if this matches an event.
    if event_name in event_names:
      if split_tag1[1] in base:
        base[event_name].append(val)
      else:
        base[event_name] = [val]
    # Check if this matches a layer event.
    if len(split_tag2) != 3:
      continue
    layer = split_tag2[1]
    layer_event_name = split_tag2[2]
    if layer_event_name in layer_event_names:
      if layer_event_name in base:
        if layer in base[layer_event_name]:
          base[layer_event_name][layer].append(val)
        else:
          base[layer_event_name][layer] = [val]
      else:
        base[layer_event_name] = {layer: [val]}
  return results

def load_all_events(filename):
  """Return a list of all events in filename."""
  return list(yield_events(filename))

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Usage: load_events.py [events file]')
    sys.exit(1)
  for e in yield_events(sys.argv[1]):
    print(e)
