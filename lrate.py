import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import superp

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


############################################
# learn rate scheduling
############################################
def set_scheduler(optimizer):
    def lr_lambda(epoch):
        rate = superp.ALPHA / (1.0 + superp.BETA * np.power(epoch, superp.GAMMA))
        return rate

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return scheduler


def reset_scheduler(optimizer):
    def lr_lambda(epoch):
        rate = superp.ALPHA / superp.SHRINK_RATE_FACTOR / (1.0 + superp.BETA * np.power(epoch, superp.GAMMA))
        return rate

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return scheduler