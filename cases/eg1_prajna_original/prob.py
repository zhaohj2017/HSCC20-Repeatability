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
INIT = [[1, 2], \
            [-0.5, 0.5], \
        ]
INIT_SHAPE = 2 # 2 for circle

SUB_INIT = []
SUB_INIT_SHAPE = []

# the the unsafe in super-rectangle
UNSAFE = [[-1.4, -0.6], \
            [-1.4, -0.6], \
        ]
UNSAFE_SHAPE = 2 # 2 for circle

SUB_UNSAFE = []
SUB_UNSAFE_SHAPE = []

# the the domain in super-rectangle
DOMAIN = [[-3, 2.5], \
            [-2, 1], \
        ]
DOMAIN_SHAPE = 1 # 1 for rectangle

############################################
# set the range constraints
############################################
# accept a two-dimensional tensor and return a 
# tensor of bool with the same number of columns
def cons_init(x): 
    return torch.pow(x[:, 0] - 1.5, 2) + \
            torch.pow(x[:, 1], 2) <= 0.25 + \
               superp.TOL_DATA_GEN 
    # x[:, 0] stands for x1 and x[:, 1] stands for x2

def cons_unsafe(x):
    return torch.pow(x[:, 0] + 1, 2) + torch.pow(x[:, 1] + 1, 2) <= 4 / 25.0 + superp.TOL_DATA_GEN 
    # x[:, 0] stands for x1 and x[:, 1] stands for x2

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
            return x[:, 1] # x[:, 1] stands for x2
        elif i == 2:
            return - x[:, 0] - x[:, 1] + \
                torch.pow(x[:, 0], 3) / 3.0 
                # x[:, 0] stands for x1
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([f(i + 1, x) for i in range(DIM)], dim=1)
    return vf
