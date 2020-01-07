import torch
import torch.nn as nn
import numpy as np
import superp # parameters
import ann # generating model
import data # generating data
import loss # computing loss
import opt
import lrate


############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


#################################################
# iterative training: the most important function
# it relies on three assistant functions:
# initialize_p, half_p, and closure
#################################################
def itr_train(model, batches_init, batches_unsafe, batches_domain):
    # set the number of restart times
    num_restart = -1

    # initialize optimizer and scheduler
    optimizer = None
    scheduler = None

    epoch_gradient_flag = True
  
    # used to output learned model parameters
    def print_nn():
        for p in model.parameters():
            print(p.data)


    # used for initialization and restart
    def initialize_p():        
        nonlocal model
        nonlocal optimizer
        nonlocal scheduler

        with torch.no_grad():
            if superp.FINE_TUNE == 0:
                    print("Initialize parameters!")
                    print("")

                    for p in model.parameters():
                        nn.init.normal_(p) #standard Gaussian distribution
            else:
                model.load_state_dict(torch.load('pre-trained.pt'), strict=True)

        optimizer = opt.set_optimizer(model)
        scheduler = lrate.set_scheduler(optimizer)

    # half_p() is to prevent generating a nn with large gradient, it works only for nn model with a single hidden layer
    def half_p():
        nonlocal model
        nonlocal optimizer
        nonlocal scheduler
        nonlocal reset_optimizer_flag

        with torch.no_grad():
            print("Half parameters!")
            i = 0
            for p in model.parameters(): # i = 1, 3, 5, 7: weight matrix; i = 2, 4, 6, 8: bias
                i = i + 1
                if i % 2 == 0:
                    p.data = p.data / torch.pow(torch.tensor(2), i // 2)
                else:
                    p.data = p.data / 2
                    # for two hidden layeres, divided by 1.6???
        
        if reset_optimizer_flag:
            optimizer = opt.reset_optimizer(model)
            scheduler = lrate.reset_scheduler(optimizer)
        else:
            optimizer = opt.set_optimizer(model)
            scheduler = lrate.set_scheduler(optimizer)


    # mini-batch training
    def train_mini_batch(batch_init, batch_unsafe, batch_domain):
        batch_gradient_flag = True
        curr_batch_loss = 0

        def closure(): # closure() called in optimizer
            nonlocal curr_batch_loss
            nonlocal batch_gradient_flag

            optimizer.zero_grad() # clear gradient of parameters

            batch_loss, batch_gradient = loss.calc_loss(model, batch_init, batch_unsafe, batch_domain) 
                # batch_loss is a tensor, batch_gradient is a scalar

            curr_batch_loss = batch_loss.item()
            batch_gradient_flag = batch_gradient < superp.TOL_MAX_GRAD # update gradient flag

            if superp.VERBOSE == 1:
                print("restart:", num_restart, "epoch:", epoch, "batch:", batch_index, "batch_loss:", batch_loss.item(), \
                        "batch_gradient:", batch_gradient, "epoch_loss:", epoch_loss, "\n")

            # sometimes we need to set the last gradient to zero
            # so optimizer.step() will not update the parameters
            if batch_gradient_flag:
                batch_loss = batch_loss
            else:
                batch_loss = 0 * batch_loss # if gradient is too large, parameters will be halfed rather than updated using gradient
                
            batch_loss.backward() # compute gradient using backward()

            return batch_loss

        optimizer.step(closure) # gradient descent once
        scheduler.step() # re-schedule learning rate once

        if not batch_gradient_flag: # if gradient is too large, parameters will be halfed rather than updated using gradient
            half_p()

        return curr_batch_loss, batch_gradient_flag


    # iterative training
    while num_restart < 5:
        num_restart += 1
        initialize_p() # restart by re-initializing nn parameters
        
        # for precise tuning of each batch
        reset_optimizer_flag = False
        num_batch_itr = 1

        init_list = np.arange(superp.BATCHES) % superp.BATCHES_I  # generate batch indices
        unsafe_list = np.arange(superp.BATCHES) % superp.BATCHES_U
        domain_list = np.arange(superp.BATCHES) % superp.BATCHES_D

        for epoch in range(superp.EPOCHS): # train for a number of epochs
            # initialize epoch
            epoch_loss = 0 # scalar
            epoch_gradient_flag = True

            # mini-batches shuffle by shuffling batch indices
            np.random.shuffle(init_list) 
            np.random.shuffle(unsafe_list)
            np.random.shuffle(domain_list)

            num_instable_batch = 0
            # train mini-batches
            for batch_index in range(superp.BATCHES):
                if reset_optimizer_flag:
                    # resest optimizer (to LBFGS or other) and scheduler for this batch
                    optimizer = opt.reset_optimizer(model)
                    scheduler = lrate.reset_scheduler(optimizer)

                # batch data selection
                batch_init = batches_init[init_list[batch_index]]
                batch_unsafe = batches_unsafe[unsafe_list[batch_index]]
                batch_domain = batches_domain[domain_list[batch_index]]

                # batch train
                for batch_itr in range(num_batch_itr):
                    curr_batch_loss, curr_batch_gradient_flag = train_mini_batch(batch_init, batch_unsafe, batch_domain)
                    if curr_batch_loss > superp.LOSS_OPT_FLAG:
                        num_instable_batch += 1
                    else:
                        break
                        
                # update epoch_loss
                epoch_loss += curr_batch_loss
                # update epoch gradient flag
                epoch_gradient_flag = epoch_gradient_flag and curr_batch_gradient_flag

            if epoch_loss < superp.LOSS_OPT_FLAG and epoch_gradient_flag:
                print("The last epoch:", epoch)
                break # epoch success: end of epoch training

            pre_reset_flag = reset_optimizer_flag
            reset_optimizer_flag = (epoch_loss < superp.TOL_OPTIMIZER_RESET) or (num_instable_batch < superp.BATCHES // superp.FRACTION_INSTABLE_BATCH)
            if (not pre_reset_flag) and reset_optimizer_flag:
                print("Reset optimizer and scheduler!")
                num_batch_itr = superp.NUM_BATCH_ITR
            if pre_reset_flag and (not reset_optimizer_flag): # restore optimizer to SGD
                print("Restore to SGD!")
                num_batch_itr = 1
                # optimizer = opt.set_optimizer(model)
                # scheduler = lrate.set_scheduler(optimizer)
                initialize_p()

        if epoch_loss < superp.LOSS_OPT_FLAG and epoch_gradient_flag:
            print("Success! The nn model is:\n")
            print_nn() # output the learned model
            break # Success: end of restart while loop

    # return the learned model
    print("restart:", num_restart)
    return model