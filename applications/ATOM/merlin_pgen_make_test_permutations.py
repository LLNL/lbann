from maestrowf.datastructures.core import ParameterGenerator
import itertools as iter

def split(word):
    return list(word)

def get_custom_generator(env, **kwargs):
    p_gen = ParameterGenerator()

    # Unpack any pargs passed in
    num_nodes = kwargs.get('num_nodes', '1').split(" ")
    mb_size = kwargs.get('mb_size', '1').split(" ")
    procs_per_trainer = kwargs.get('ppt', '1').split(" ")
    iter_per_tournament = kwargs.get('ipt', '1')

    num_nodes_values = []
    mb_sizes_values = []
    procs_per_trainer_values = []
    ltfbbi_values = []
    trial_values = []

    for trial, param_combo in enumerate(iter.product(num_nodes,mb_size,procs_per_trainer)):
        num_nodes_values.append(param_combo[0])
        mb_sizes_values.append(param_combo[1])
        procs_per_trainer_values.append(param_combo[2])
        ltfbbi_values.append(int(int(iter_per_tournament) / int(param_combo[1])))
        trial_values.append(trial)

    params = {
        "0TRIAL": {
            "values": trial_values,
            "label": "t%%"
        },
        "1NUM_NODES": {
            "values": num_nodes_values,
            "label": "n%%"
        },
        "2BATCH_SIZE": {
            "values": mb_sizes_values,
            "label": "bs%%"
        },
        "3PPT": {
            "values": procs_per_trainer_values,
            "label": "ppt%%"
        },
        "4LTFBBI": {
            "values": ltfbbi_values,
            "label": "LTFBbi%%"
        },
    }

    for key in sorted(params):
        value = params[key]
        p_gen.add_parameter(key, value["values"], value["label"])

    return p_gen
