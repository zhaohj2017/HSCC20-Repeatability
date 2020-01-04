import torch
import torch.nn as nn
import superp
import prob

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


############################################
# constraints for barrier certificate B:
# (1) init ==> B <= 0
# (2) unsafe ==> B > 0 <==> B >= eps <==> eps - B <= 0 (positive eps)
# (3) domain ==> lie <= 0  (lie + lambda * barrier <= 0, where lambda >= 0)
# (4) domain /\ B = 0 ==> lie < 0 (alternatively)
############################################


############################################
# given the training data, compute the loss
############################################
def calc_loss(model, input_init, input_unsafe, input_domain):
    # compute the output of nn on domain
    input_domain.requires_grad = True # temporarily enable gradient
    output_domain = model(input_domain)
    print(output_domain)
    input()

    # compute the gradient of nn on domain
    gradient_domain = torch.autograd.grad(
            list(output_domain),
            input_domain,
            grad_outputs=None,
            create_graph=True,
            only_inputs=True,
            allow_unused=True)[0]
    input_domain.requires_grad = False # temporarily disable gradient
    
    # compute the vector field on domain
    vector_domain = prob.vector_field(input_domain) # with torch.no_grad():
  
  # compute the norm of vector field
    norm_vector = torch.norm(vector_domain, dim=1)
  # compute the norm of gradient
    norm_gradient = torch.norm(gradient_domain, dim=1)
    with torch.no_grad():
        max_gradient = torch.max(norm_gradient) # computing max norm of gradient for controlling boundary sampling
    # norm_gradient[norm_gradient < superp.LOSS_OPT_FLAG] = 1e16 # avoid zero gradient for computing norm_lie
        # A possible bug here! norm_gradient = 0?

    # compute the lie derivative on domain
    lie = torch.sum(gradient_domain * vector_domain, dim=1) # sum the columns of lie_domain
    # compute normalized lie
    norm_lie = lie / norm_vector / norm_gradient

    with torch.no_grad():
        boundary_index = ((output_domain[:,0] >= -superp.TOL_BOUNDARY) & (output_domain[:,0] <= superp.TOL_BOUNDARY)).nonzero()
                #torch.index_select(a, 0, b[:,0])

    boundary_lie = torch.index_select(lie, 0, boundary_index[:, 0])
    boundary_norm_lie = torch.index_select(norm_lie, 0, boundary_index[:, 0])

    # compute loss of lie
    if len(boundary_lie) == 0:
        loss_lie = torch.tensor(0) * output_domain # in order to call loss.backwar()
        # avg_loss_lie = torch.tensor(0) * output_domain[0, 0]
    else:
        loss_lie = superp.WEIGHT_NORM_LIE * torch.relu(boundary_norm_lie + superp.TOL_NORM_LIE) + superp.WEIGHT_LIE * torch.relu(boundary_lie + superp.TOL_LIE)
        # avg_loss_lie = torch.sum(loss_lie) / loss_lie.size()[0]

    # compute loss of init    
    if len(input_init) == 0:
        loss_init = torch.tensor(0) * output_domain # in order to call loss.backwar()
        # avg_loss_init = torch.tensor(0) * output_domain[0, 0]
    else:
        output_init = model(input_init)
        loss_init = torch.relu(output_init + superp.TOL_INIT) #tolerance
        # avg_loss_init = torch.sum(loss_init) / loss_init.size()[0]


    # compute loss of unsafe
    if len(input_unsafe) == 0:
        loss_unsafe = torch.tensor(0) * output_domain # in order to call loss.backwar()
        # avg_loss_unsafe = torch.tensor(0) * output_domain[0, 0]
    else:
        output_unsafe = model(input_unsafe)
        loss_unsafe = torch.relu((- output_unsafe) + superp.TOL_SAFE) #tolerance
        # avg_loss_unsafe = torch.sum(loss_unsafe) / loss_unsafe.size()[0]
    
    # compute total loss
    total_loss = superp.DECAY_LIE * torch.sum(loss_lie) + superp.DECAY_INIT * torch.sum(loss_init) + superp.DECAY_UNSAFE * torch.sum(loss_unsafe)
    # avg_total_loss = superp.DECAY_LIE * avg_loss_lie + superp.DECAY_INIT * avg_loss_init + superp.DECAY_UNSAFE * avg_loss_unsafe

    # return avg_total_loss, max_gradient.item() # total_loss is a tensor, max_gradient is a scalar
    return total_loss, max_gradient.item() 