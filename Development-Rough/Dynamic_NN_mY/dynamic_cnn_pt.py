import torch
import torch.nn as nn
import numpy as np

def _get_hidden_neuron_number(i, o):
    
    #### get minimum of any
#     return min((i, o))+1

    #### get geometric mean
#     return np.sqrt(i*o)

    #### modified geometric mean
    return (max(i,o)*(min(i,o)**2))**(1/3)


