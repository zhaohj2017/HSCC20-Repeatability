import torch
import superp

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


############################################
# set the system dimension
############################################
DIM = 2


############################################
# set the super-rectangle range
############################################
# set the initial in super-rectangle
INIT = [[-0.9, -0.1], \
            [0.1, 0.9] \
        ]

INIT_SHAPE = 2 # 2 for circle

SUB_INIT = []
SUB_INIT_SHAPE = []


# the the unsafe in super-rectangle
UNSAFE = [[0.4, 1.0], \
            [-1.0, -0.4] \
        ]

UNSAFE_SHAPE = 2 # 2 for circle

SUB_UNSAFE = [
]

SUB_UNSAFE_SHAPE = []


# the the domain in super-rectangle
DOMAIN = [[-2, 2], \
            [-2, 2], \
        ]

DOMAIN_SHAPE = 1 # 1 for rectangle

############################################
# set the range constraints
############################################
def cons_init(x): # accept a two-dimensional tensor and return a tensor of bool with the same number of columns
    return torch.pow(x[:, 0] + 0.5, 2) + torch.pow(x[:, 1] - 0.5, 2) <= 0.16 + superp.TOL_DATA_GEN

def cons_unsafe(x):
    return torch.pow(x[:, 0] - 0.7, 2) + torch.pow(x[:, 1] + 0.7, 2) <= 0.09 + superp.TOL_DATA_GEN 

def cons_domain(x):
    return x[:, 0] == x[:, 0] # equivalent to True


############################################
# set the vector field
############################################
# this function accepts a tensor input and returns the vector field of the same size
def vector_field(x):
    # the vector of functions
    def f(i, x):
        if i == 1:
            return torch.exp(-x[:, 0]) + x[:, 1] - 1 # x[:, 1] stands for x2
        elif i == 2:
            return -1 * torch.sin(x[:, 0]) * torch.sin(x[:, 0]) # x[:, 0] stands for x1

        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([f(i + 1, x) for i in range(DIM)], dim=1)

    return vf
