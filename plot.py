import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import prob
import superp
import plot3d

############################################
# This file will plot 2-D or 3-D barrier
############################################


############################################
# set default data type to double
############################################
# torch.set_default_dtype(torch.float64)
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)


############################################
# plot function for 2d systems
############################################    
def plot_barrier_2d(model):
    # sampling data for plotting barrier, vector field and scattering sample points
    def gen_plot_data(region, len_sample):
        grid_sample = [torch.linspace(region[i][0], region[i][1], int(len_sample[i])) for i in range(prob.DIM)] # gridding each dimension
        mesh = torch.meshgrid(grid_sample) # mesh the gridding of each dimension
        flatten = [torch.flatten(mesh[i]) for i in range(len(mesh))] # flatten the list of meshes
        plot_data = torch.stack(flatten, 1) # stack the list of flattened meshes
        return plot_data

    def plot_boundary(): # barrier boundary: contour plotting
        barrier_plot_nn_input = gen_plot_data(prob.DOMAIN, superp.PLOT_LEN_B)
        # apply the nn model but do not require gradient
        with torch.no_grad():
            barrier_plot_nn_output = model(barrier_plot_nn_input).reshape(superp.PLOT_LEN_B[1], superp.PLOT_LEN_B[0]) # y_size * x_size
        plot_Z = barrier_plot_nn_output.numpy()
        plot_sample_x = np.linspace(prob.DOMAIN[0][0], prob.DOMAIN[0][1], superp.PLOT_LEN_B[0])
        plot_sample_y = np.linspace(prob.DOMAIN[1][0], prob.DOMAIN[1][1], superp.PLOT_LEN_B[1])
        plot_X, plot_Y = np.meshgrid(plot_sample_x, plot_sample_y)
        #plt.contourf(plot_X, plot_Y, plot_Z, [0], color='k')
        barrier_contour = plt.contour(plot_X.T, plot_Y.T, plot_Z, [-superp.TOL_BOUNDARY, 0, superp.TOL_BOUNDARY], \
            linewidths = 3, colors=('k', 'b', 'y'))
        plt.clabel(barrier_contour, fontsize=20, colors=('k', 'b', 'y'))
        return barrier_contour

    def plot_scatter(): # scatterring sample points
        scatter_plot_nn_input = gen_plot_data(prob.DOMAIN, superp.PLOT_LEN_P)
        x_values = (scatter_plot_nn_input[:, 0]).numpy()
        y_values = (scatter_plot_nn_input[:, 1]).numpy()
        scattering_points = plt.scatter(x_values, y_values)
        return scattering_points

    def plot_vector_field(): # vector field
        vector_plot_nn_input = gen_plot_data(prob.DOMAIN, superp.PLOT_LEN_V)
        vector_field = prob.vector_field(vector_plot_nn_input)

        vector_x_values = (vector_field[:, 0]).numpy()
        vector_y_values = (vector_field[:, 1]).numpy()
        vector_x_positions = (vector_plot_nn_input[:, 0]).numpy()
        vector_y_positions = (vector_plot_nn_input[:, 1]).numpy()

        vector_plot = plt.quiver(vector_x_positions, vector_y_positions, vector_x_values, vector_y_values, \
                        color='pink', angles='xy', scale_units='xy', scale=superp.PLOT_VEC_SCALE)
        return vector_plot

    def plot_init(init_range, init_shape):
        if init_shape == 1: # rectangle
            init = matplotlib.patches.Rectangle((init_range[0][0], init_range[1][0]), \
                init_range[0][1] - init_range[0][0], init_range[1][1] - init_range[1][0], facecolor='green')
        if init_shape == 2: # circle
            init = matplotlib.patches.Circle(((init_range[0][1] +  init_range[0][0]) / 2.0, \
                    (init_range[1][1] + init_range[1][0]) / 2.0), (init_range[1][1] - init_range[1][0]) / 2.0, facecolor='green')
        return init

    def plot_unsafe(unsafe_range, unsafe_shape):
        if unsafe_shape == 1: # rectangle
            unsafe = matplotlib.patches.Rectangle((unsafe_range[0][0], unsafe_range[1][0]), \
                unsafe_range[0][1] - unsafe_range[0][0], unsafe_range[1][1] - unsafe_range[1][0], facecolor='red')
        elif unsafe_shape == 2: # circle
            unsafe = matplotlib.patches.Circle(((unsafe_range[0][1] + unsafe_range[0][0]) / 2.0, \
                    (unsafe_range[1][1] + unsafe_range[1][0]) / 2.0), (unsafe_range[1][1] - unsafe_range[1][0]) / 2.0, facecolor='red')
        else: # a parabola?
            y = np.linspace(-np.sqrt(2), np.sqrt(2), 1000)
            x = - y ** 2
            unsafe = plt.fill(x, y, 'r')

        return unsafe
    
    fig, ax = plt.subplots()
    boundary = plot_boundary() # plot boundary of barrier function
    scatter = plot_scatter() # plot scatter points
    vector_field = plot_vector_field() # plot vector field
    
    # plot sub_init
    if len(prob.SUB_INIT) == 0:
        init = plot_init(prob.INIT, prob.INIT_SHAPE) # plot initial
        ax.add_patch(init)
    else:
        for i in range(len(prob.SUB_INIT)):
            init = plot_init(prob.SUB_INIT[i], prob.SUB_INIT_SHAPE[i]) # plot initial
            ax.add_patch(init)

    # plot sub_unsafe
    if len(prob.SUB_UNSAFE) == 0:
        unsafe = plot_unsafe(prob.UNSAFE, prob.UNSAFE_SHAPE) # plot unsafe
        if prob.UNSAFE_SHAPE == 1 or prob.UNSAFE_SHAPE == 2:
            ax.add_patch(unsafe)
    else:
        for i in range(len(prob.SUB_UNSAFE)):
            unsafe = plot_unsafe(prob.SUB_UNSAFE[i], prob.SUB_UNSAFE_SHAPE[i]) # plot unsafe
            ax.add_patch(unsafe)

    plt.axis([prob.DOMAIN[0][0], prob.DOMAIN[0][1], prob.DOMAIN[1][0], prob.DOMAIN[1][1]])
    # plt.axis('equal') ##PLOT_VEC_SCALE = None
    plt.show()


############################################
# the main plot function
############################################
def plot_barrier(model):
    if prob.DIM == 2:
        plot_barrier_2d(model)
    elif prob.DIM == 3:
        plot3d.plot_barrier_3d(model)
    else:
        print("Barrier can only be displayed for 2d or 3d systems!")

