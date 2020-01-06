import torch
import torch.nn as nn
import torch.optim as optim
import superp

############################################
# set default data type to double
############################################
# torch.set_default_dtype(torch.float64)
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)


def set_optimizer(model):
    
    # SGD
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0)

    ## LBFGS
    # optimizer = optim.LBFGS(model.parameters(), max_iter=superp.LBFGS_NUM_ITER, tolerance_grad=superp.LBFGS_TOL_GRAD, tolerance_change=superp.LBFGS_TOL_CHANGE, history_size=superp.LBFGS_NUM_HISTORY, line_search_fn=superp.LBFGS_LINE_SEARCH_FUN)
    
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

    # optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

    # optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # optimizer = optim.Adamax(model.parameters()(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

    # optimizer = optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))

    # optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    ## only applicable to sparse gradient
    # optimizer = optim.SparseAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    return optimizer

def reset_optimizer(model):

    ## LBFGS
    optimizer = optim.LBFGS(model.parameters(), max_iter=superp.LBFGS_NUM_ITER, tolerance_grad=superp.LBFGS_TOL_GRAD, tolerance_change=superp.LBFGS_TOL_CHANGE, history_size=superp.LBFGS_NUM_HISTORY, line_search_fn=superp.LBFGS_LINE_SEARCH_FUN)

    # SGD
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # ADAM
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    return optimizer
