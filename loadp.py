import torch
import torch.nn as nn
import ann
import plot
import data
import loss

############################################
# set default data type to double
############################################
# torch.set_default_dtype(torch.float64)
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)


model = ann.gen_nn()


############################################
# the learned parameters can be found in 
# ./examples/[eg-name]/learn_res.txt
############################################
list_p = [
torch.tensor([[0.1,  0.3],
        [ 0.2, 0.4]
]),
torch.tensor([0.5, 0.6]),
torch.tensor([[0.7, 0.8]]),
torch.tensor([0.9])
]

# load parameters
i = 0
for p in model.parameters():
    p.data = list_p[i]
    i = i + 1

# # for debugging
# debug_input = torch.tensor([[1.0, 0.0]])
# loss.calc_loss(model, debug_input, debug_input, debug_input)

plot.plot_barrier(model)
