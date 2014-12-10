-----------------------------
Installation and Requirements
-----------------------------

--- GPU and CUDA:

To run the GPU version of the simulation, you need to have a CUDA capable nVidia graphics card. All compute capabalities should be supported, that is, any CUDA capable nVidia card should be able to run the simulation.

You need to have the CUDA compiler (nvcc) installed and configured correctly on your system. The simulation has only been tested on CUDA 5.0 and should run on newer versions. It might also run on older versions of CUDA.

--- Python:

You need to have Python 2.7 installed. The simulation does not support python 3.

Required packages are: PyCUDA, NumPy.

For the visual simulations, you also need Pygame.

------
Usage
------

There are 4 Python scripts. 2 that run on the CPU and 2 that run on the GPU. For both GPU and CPU, there is a script that just runs the simulation and one that runs it and simultaneously visualizes the simulation in a Pygame window.

The simulation is visualized as each node in the simulation mapped to an area of onscreen pixels (by default, each node is 2*2 pixels). The pixels are colored (red for the CPU version and green for the GPU version) by the absolute velocity of the fluid in the node. Higher intensity means higher velocity. Solid nodes are black (as well as fluid nodes with 0 velocity).

the simulation takes 3 arguments, these are the x and y dimensions of the simulation and the number of iterations.

example:

python lbgk_GPU.py 1024 256 1000

---------------------------------------
Notes on Performance and Implementation
---------------------------------------

--- The CPU version:

The CPU version of the simulation was written with almost no thought to performance, as it was written primarily as an exercise for me on how to implement LBM sequentially, to prepare for the CUDA implementation. As a result, it is very slow.

--- Visualization:

The simulation has been visualized with the Pygame module. This is a very simple way of visualization and it slows down the simulation significantly for the GPU simulation, as it involves copying memory back and forth between main memory and GPU memory, every time a new frame should be drawn. Ideally, OpenGL should be used to visualize the simulation, to avoid the memory copies. The Pygame method works fine for illustrative purposes however.

---------------
Science-y stuff
---------------

The simulation was written with the 2DQ9 scheme of LBM. 2D was chosen, since 3D would take up an enormous amount of memory compared to 2D. Way too much to fit in the GPU memory, so another buffering method would have to be devised. The Bhatnager-Gross-Krook collision model was chosen as it seems to be the most used model and it is easily implemented.

The reason for the choice of geometry is that this geometry clearly demonstrates the Kármán Vortex Street for appropiate fluid parameters, which was part of my motivation for doing the project. The simulation should be fairly easily converted to other geometries however, since one of the strenghts of LBM is that the basic implementation is fairly independent on geometry.