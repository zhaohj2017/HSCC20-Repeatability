import torch
import torch.nn as nn
import numpy as np
from mayavi import mlab
from mayavi.mlab import *
import superp
import prob


############################################
# set default data type to double
############################################
# torch.set_default_dtype(torch.float64)
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)

###########################################
# plot function for 3-d systems
# support cube, sphere and cylinder
###########################################


# code for plotting cube: from
# https://stackoverflow.com/questions/26888098/draw-cubes-with-colour-intensity-with-python
def mlab_plt_cube(xmin, xmax, ymin, ymax, zmin, zmax, c_color):
	def cube_faces(xmin, xmax, ymin, ymax, zmin, zmax):
		faces = []

		x,y = np.mgrid[xmin:xmax:3j,ymin:ymax:3j]
		z = np.ones(y.shape)*zmin
		faces.append((x,y,z))

		x,y = np.mgrid[xmin:xmax:3j,ymin:ymax:3j]
		z = np.ones(y.shape)*zmax
		faces.append((x,y,z))

		x,z = np.mgrid[xmin:xmax:3j,zmin:zmax:3j]
		y = np.ones(z.shape)*ymin
		faces.append((x,y,z))

		x,z = np.mgrid[xmin:xmax:3j,zmin:zmax:3j]
		y = np.ones(z.shape)*ymax
		faces.append((x,y,z))

		y,z = np.mgrid[ymin:ymax:3j,zmin:zmax:3j]
		x = np.ones(z.shape)*xmin
		faces.append((x,y,z))

		y,z = np.mgrid[ymin:ymax:3j,zmin:zmax:3j]
		x = np.ones(z.shape)*xmax
		faces.append((x,y,z))

		return faces
		
	def trans_x(origin_x):
		k = (superp.PLOT_LEN_B[0] - 1) / (prob.DOMAIN[0][1] - prob.DOMAIN[0][0])
		c = 1 - k * prob.DOMAIN[0][0]
		return k * origin_x + c

	def trans_y(origin_y):
		k = (superp.PLOT_LEN_B[1] - 1) / (prob.DOMAIN[1][1] - prob.DOMAIN[1][0])
		c = 1 - k * prob.DOMAIN[1][0]
		return k * origin_y + c

	def trans_z(origin_z):
		k = (superp.PLOT_LEN_B[2] - 1) / (prob.DOMAIN[2][1] - prob.DOMAIN[2][0])
		c = 1 - k * prob.DOMAIN[2][0]
		return k * origin_z + c

	faces = cube_faces( trans_x(xmin), trans_x(xmax), trans_y(ymin), trans_y(ymax), trans_z(zmin), trans_z(zmax) )
	for grid in faces:
		x,y,z = grid
		mlab.mesh(x,y,z,color=c_color,opacity=0.3)


# plot sphere
def mlab_plt_sphere(center_x, center_y, center_z, s_rad, s_color):
	x, y, z = np.ogrid[ (prob.DOMAIN[0][0]) : (prob.DOMAIN[0][1]) : complex(0, superp.PLOT_LEN_B[0]), \
							(prob.DOMAIN[1][0]) : (prob.DOMAIN[1][1]) : complex(0, superp.PLOT_LEN_B[1]), \
								(prob.DOMAIN[2][0]) : (prob.DOMAIN[2][1]) : complex(0, superp.PLOT_LEN_B[2]) ]

	sphere_scalar = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) + (z - center_z) * (z - center_z)
	sphere = mlab.contour3d(sphere_scalar, contours = [s_rad * s_rad], color = s_color, opacity = 0.3)


# plot sphere
def mlab_plt_cylinder(center_x, center_y, s_rad, s_color):
	x, y, z = np.ogrid[ (prob.DOMAIN[0][0]) : (prob.DOMAIN[0][1]) : complex(0, superp.PLOT_LEN_B[0]), \
							(prob.DOMAIN[1][0]) : (prob.DOMAIN[1][1]) : complex(0, superp.PLOT_LEN_B[1]), \
								(prob.DOMAIN[2][0]) : (prob.DOMAIN[2][1]) : complex(0, superp.PLOT_LEN_B[2]) ]

	cylinder_scalar = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) - 0 * z
	cylinder = mlab.contour3d(cylinder_scalar, contours = [s_rad * s_rad], color = s_color, opacity = 0.3)


# plot initial
def mlab_plt_init():
	if prob.INIT_SHAPE == 1: # cube
		mlab_plt_cube(prob.INIT[0][0], prob.INIT[0][1], prob.INIT[1][0], prob.INIT[1][1], prob.INIT[2][0], prob.INIT[2][1], (0, 1, 0) )
	else: # sphere
		mlab_plt_sphere( (prob.INIT[0][0] + prob.INIT[0][1]) / 2, (prob.INIT[1][0] + prob.INIT[1][1]) / 2, \
							(prob.INIT[2][0] + prob.INIT[2][1]) / 2, (prob.INIT[0][1] - prob.INIT[0][0]) / 2, (0, 1, 0) )


# plot unsafe
def mlab_plt_unsafe():
	if len(prob.SUB_UNSAFE) == 0:
		if prob.UNSAFE_SHAPE == 1: # cube
			mlab_plt_cube(prob.UNSAFE[0][0], prob.UNSAFE[0][1], prob.UNSAFE[1][0], prob.UNSAFE[1][1], prob.UNSAFE[2][0], prob.UNSAFE[2][1], (1, 0, 0) )
		elif prob.UNSAFE_SHAPE == 2: # sphere
			mlab_plt_sphere( (prob.UNSAFE[0][0] + prob.UNSAFE[0][1]) / 2, (prob.UNSAFE[1][0] + prob.UNSAFE[1][1]) / 2, \
								(prob.UNSAFE[2][0] + prob.UNSAFE[2][1]) / 2, (prob.UNSAFE[0][1] - prob.UNSAFE[0][0]) / 2, (1, 0, 0) )
		elif prob.UNSAFE_SHAPE == 3: # cylinder
			mlab_plt_cylinder( (prob.UNSAFE[0][0] + prob.UNSAFE[0][1]) / 2, (prob.UNSAFE[1][0] + prob.UNSAFE[1][1]) / 2, \
				(prob.UNSAFE[0][1] - prob.UNSAFE[0][0]) / 2, (1, 0, 0) )
		else:
			x, y, z = np.ogrid[ (prob.DOMAIN[0][0]) : (prob.DOMAIN[0][1]) : complex(0, superp.PLOT_LEN_B[0]), \
							(prob.DOMAIN[1][0]) : (prob.DOMAIN[1][1]) : complex(0, superp.PLOT_LEN_B[1]), \
								(prob.DOMAIN[2][0]) : (prob.DOMAIN[2][1]) : complex(0, superp.PLOT_LEN_B[2]) ]
			hyperplane_scalar = x + y + z
			hyperplane = mlab.contour3d(hyperplane_scalar, contours = [1], color = (1, 0, 0), opacity = 0.3)

	else:
		for i in range(len(prob.SUB_UNSAFE)):
			curr_shape = prob.SUB_UNSAFE_SHAPE[i]
			curr_range = prob.SUB_UNSAFE[i]
			if curr_shape == 1: # cube
				mlab_plt_cube(curr_range[0][0], curr_range[0][1], curr_range[1][0], curr_range[1][1], curr_range[2][0], curr_range[2][1], (1, 0, 0) )
			elif curr_shape == 2: # sphere
				mlab_plt_sphere( (curr_range[0][0] + curr_range[0][1]) / 2, (curr_range[1][0] + curr_range[1][1]) / 2, \
									(curr_range[2][0] + curr_range[2][1]) / 2, (curr_range[0][1] - curr_range[0][0]) / 2, (1, 0, 0) )
			else : # cylinder
				mlab_plt_cylinder( (curr_range[0][0] + curr_range[0][1]) / 2, (curr_range[1][0] + curr_range[1][1]) / 2, \
									 (curr_range[0][1] - curr_range[0][0]) / 2, (1, 0, 0) )


# generating plot data for nn
def gen_plot_data():
	sample_x = torch.linspace(prob.DOMAIN[0][0], prob.DOMAIN[0][1], int(superp.PLOT_LEN_B[0]))
	sample_y = torch.linspace(prob.DOMAIN[1][0], prob.DOMAIN[1][1], int(superp.PLOT_LEN_B[1]))
	sample_z = torch.linspace(prob.DOMAIN[2][0], prob.DOMAIN[2][1], int(superp.PLOT_LEN_B[2]))
	grid_xyz = torch.meshgrid([sample_x, sample_y, sample_z])
	flatten_xyz = [torch.flatten(grid_xyz[i]) for i in range(len(grid_xyz))]
	plot_input = torch.stack(flatten_xyz, 1)
	
	return plot_input


# plot barrier
def mlab_plt_barrier(model):
	# generating nn_output for plotting
	plot_input = gen_plot_data()
	nn_output = model(plot_input)
	plot_output = (nn_output[:, 0]).reshape(superp.PLOT_LEN_B[0], superp.PLOT_LEN_B[1], superp.PLOT_LEN_B[2])
	# barrier_plot = mlab.contour3d(plot_output.detach().numpy(), contours = [-superp.TOL_BOUNDARY, 0, -superp.TOL_BOUNDARY], \
	# 								color = (1, 1, 0), opacity = 0.3) # yellow
	src = mlab.pipeline.scalar_field(plot_output.detach().numpy())
	barrier_plot = mlab.pipeline.iso_surface(src, contours = [-superp.TOL_BOUNDARY, 0, superp.TOL_BOUNDARY], \
									color = (1, 1, 0), opacity = 1)


# plot vector field
def mlab_plt_vector():
	vector_input = gen_plot_data()
	vector_field = prob.vector_field(vector_input)
	# attention! for plot-3d to work, superp.PLOT_LEN_V should be equal to superp.PLOT_LEN_B
	u = (vector_field[:, 0]).reshape(superp.PLOT_LEN_B[0], superp.PLOT_LEN_B[1], superp.PLOT_LEN_B[2])
	v = (vector_field[:, 1]).reshape(superp.PLOT_LEN_B[0], superp.PLOT_LEN_B[1], superp.PLOT_LEN_B[2])
	w = (vector_field[:, 2]).reshape(superp.PLOT_LEN_B[0], superp.PLOT_LEN_B[1], superp.PLOT_LEN_B[2])

	mlab.quiver3d(u.numpy(), v.numpy(), w.numpy(), color = (0, 0, 1), mask_points = 10, scale_factor = 0.1)


# plot vector field
def mlab_plt_flow():
	vector_input = gen_plot_data()
	vector_field = prob.vector_field(vector_input)
	# attention! for plot-3d to work, superp.PLOT_LEN_V should be equal to superp.PLOT_LEN_B
	u = (vector_field[:, 0]).reshape(superp.PLOT_LEN_B[0], superp.PLOT_LEN_B[1], superp.PLOT_LEN_B[2])
	v = (vector_field[:, 1]).reshape(superp.PLOT_LEN_B[0], superp.PLOT_LEN_B[1], superp.PLOT_LEN_B[2])
	w = (vector_field[:, 2]).reshape(superp.PLOT_LEN_B[0], superp.PLOT_LEN_B[1], superp.PLOT_LEN_B[2])

	obj = flow(u, v, w, seed_scale=0.1)
	obj.stream_tracer.maximum_propagation = 500
	return obj


# plot frame
def mlab_plt_frame():
	x, y, z = np.mgrid[ (prob.DOMAIN[0][0]) : (prob.DOMAIN[0][1]) : complex(0, superp.PLOT_LEN_B[0]), \
							(prob.DOMAIN[1][0]) : (prob.DOMAIN[1][1]) : complex(0, superp.PLOT_LEN_B[1]), \
								(prob.DOMAIN[2][0]) : (prob.DOMAIN[2][1]) : complex(0, superp.PLOT_LEN_B[2]) ]
	u = 0 * x
	v = 0 * y
	w = 0 * z
	src = mlab.pipeline.vector_field(u, v, w)
	frame = mlab.pipeline.vectors(src, scale_factor = 0, opacity = 0)
	# frame = mlab.quiver3d(u, v, w, color = (1, 1, 1), scale_factor = 0, opacity = 0.0)
	# print(src.get_output_dataset())
	# input
	mlab.outline()
	mlab.axes()


# plot 3d-system
def plot_barrier_3d(model): 
	mlab.figure(fgcolor = (0, 0, 0), bgcolor = (1, 1, 1))
	mlab_plt_barrier(model)
	mlab_plt_init()
	mlab_plt_unsafe()
	mlab_plt_vector()
	mlab_plt_flow()
	mlab_plt_frame()
	mlab.show()



# ###########################################
# # plot heart
# ###########################################
# def mlab_plt_heart():
# 	x, y, z = np.ogrid[ (prob.DOMAIN[0][0]) : (prob.DOMAIN[0][1]) : complex(0, superp.PLOT_LEN_B[0]), \
# 								(prob.DOMAIN[1][0]) : (prob.DOMAIN[1][1]) : complex(0, superp.PLOT_LEN_B[1]), \
# 									(prob.DOMAIN[2][0]) : (prob.DOMAIN[2][1]) : complex(0, superp.PLOT_LEN_B[2]) ]

# 	heart_scalar = (x*x+9/4*y*y+z*z-1) * (x*x+9/4*y*y+z*z-1) * (x*x+9/4*y*y+z*z-1) - x*x*z*z*z -9/80*y*y*z*z*z
# 	heart = mlab.contour3d(heart_scalar, contours = [0], color = (1,0,0), opacity = 0.3)

