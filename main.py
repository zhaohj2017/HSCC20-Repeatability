import torch
import torch.nn as nn
import train
import plot
import ann
import data
import time

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


# generating training model
model = ann.gen_nn()

# generate training data
time_start_data = time.time()
batches_init, batches_unsafe, batches_domain = data.gen_batch_data()
time_end_data = time.time()

# train and return the learned model
time_start_train = time.time()
train.itr_train(model, batches_init, batches_unsafe, batches_domain) 
time_end_train = time.time()

print("\nData generation totally cost:", time_end_data - time_start_data)
print("Training totally cost:", time_end_train - time_start_train)

# plot
# plot.plot_barrier(model) # do not plot on gpu