import re

_parse_list = [
    ('train_loss', 'training epoch ([0-9]+) objective function : ([0-9.]+)', lambda r: float(r.group(2))),
    ('train_acc', 'training epoch ([0-9]+) categorical accuracy : ([0-9.]+)', lambda r: float(r.group(2))/100.0),
    ('train_time', 'training epoch ([0-9]+) run time : ([0-9.]+)', lambda r: float(r.group(2))),
    ('val_loss', 'validation objective function : ([0-9.]+)', lambda r: float(r.group(1))),
    ('val_acc', 'validation categorical accuracy : ([0-9.]+)', lambda r: float(r.group(1))/100.0),
    ('val_time', 'validation run time : ([0-9.]+)', lambda r: float(r.group(1))),
    ('test_loss', 'test objective function : ([0-9.]+)', lambda r: float(r.group(1))),
    ('num_procs', 'Total number of processes\s*:\s*([\d]+)', lambda r: int(r.group(1))),
    ('num_procs_on_node', 'Processes on node\s*:\s*([\d]+)', lambda r: int(r.group(1))),
]

def parse(file_path):
    """Simple regex parsing function based on LBANN outputs at current time."""
    # Store results in dict to return to user
    data_dict = dict([(p[0], []) for p in _parse_list])
    match_any = False
    with open(file_path, 'r') as fp:
        for line in fp:
            # Check all regex expressions against current line
            for field, regex, func in _parse_list:
                match = re.search(regex, line)
                # Apply processing function to needed expression if match
                if match:
                    match_any = True
                    data_dict[field].append(func(match))
    return data_dict if match_any else None
