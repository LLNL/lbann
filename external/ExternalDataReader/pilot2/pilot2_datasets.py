data_sets = {
  '3k_Disordered' : ('3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20-f20.dir', '36ebb5bbc39e1086176133c92c29b5ce'),
#  '3k_Disordered' : ('3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir', '3a5fc83d3de48de2f389f5f0fa5df6d2'),
  '3k_Ordered' : ('3k_run32_10us.35fs-DPPC.50-DOPC.10-CHOL.40.dir', '6de30893cecbd9c66ea433df0122b328'),
  '3k_Ordered_and_gel' : ('3k_run43_10us.35fs-DPPC.70-DOPC.10-CHOL.20.dir', '45b9a2f7deefb8d5b016b1c42f5fba71'),
  '6k_Disordered' : ('6k_run10_25us.35fs-DPPC.10-DOPC.70-CHOL.20.dir', '24e4f8d3e32569e8bdd2252f7259a65b'),
  '6k_Ordered' : ('6k_run32_25us.35fs-DPPC.50-DOPC.10-CHOL.40.dir', '0b3b39086f720f73ce52d5b07682570d'),
  '6k_Ordered_and_gel' : ('6k_run43_25us.35fs-DPPC.70-DOPC.10-CHOL.20.dir', '3b3e069a7c55a4ddf805f5b898d6b1d1')
  }

from collections import OrderedDict

def gen_data_set_dict():
    # Generating names for the data set
    names= {'x' : 0, 'y' : 1, 'z' : 2, 
            'CHOL' : 3, 'DPPC' : 4, 'DIPC' : 5, 
            'Head' : 6, 'Tail' : 7}
    for i in range(12):
        temp = 'BL'+str(i+1)
        names.update({temp : i+8})

    # dictionary sorted by value
    fields=OrderedDict(sorted(names.items(), key=lambda t: t[1]))

    return fields
