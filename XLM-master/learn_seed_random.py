import numpy
import torch
import random

use_cuda = torch.cuda.is_available()

def random_seeding(seed_value, use_cuda):
    numpy.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value)    # the python random
    if use_cuda: torch.cuda.manual_seed_all(seed_value) # gpu vars
