# -*- coding: utf-8 -*-
"""
..warning:: The explanations of the functions in this file and the details of
the programming methodology have been given during the lectures.
"""


# Python packages
import matplotlib.pyplot
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
import scipy.stats
import creation_peigne
import construction
# MRG packages
import zsolutions4students as zsolutions


# ..todo: Uncomment for displaying limited digits
# numpy.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def run_exercise_solution_helmholtz_dddd(prod,type='triangle',order=False):
        
    # -- set equation parameters
    wavenumber = prod*numpy.pi
    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 10, 0.0, 10
    nelemsx, nelemsy = 50, 50

    # -- generate mesh
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = zsolutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    # .. todo:: Modify the line below to define a different geometry.
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1

    # -- set boundary geometry
    # boundary composed of nodes
    # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
    nodes_on_north = zsolutions._set_square_nodes_boundary_north(node_coords)
    nodes_on_south = zsolutions._set_square_nodes_boundary_south(node_coords)
    nodes_on_east = zsolutions._set_square_nodes_boundary_east(node_coords)
    nodes_on_west = zsolutions._set_square_nodes_boundary_west(node_coords)
    nodes_on_boundary = numpy.unique(numpy.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
    # ..warning: for teaching purpose only
    # -- set exact solution
    if order==False:
        node_coords,p_elem2nodes,elem2nodes=creation_peigne.disturbed_nodes(node_coords,p_elem2nodes,elem2nodes)
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0.,1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(0.,1.)*wavenumber*complex(0.,1.)*wavenumber * solexact[i]
    # ..warning: end

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

    # ..warning: for teaching purpose only
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - (wavenumber ** 2) * solexact[i]
    # ..warning: end
    
    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = zsolutions._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M +complex(0,1)*10*M
    B = F

    # -- apply Dirichlet boundary conditions
    A, B = zsolutions._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
    sol = scipy.linalg.solve(A, B)

    # -- plot finite element solution
    solreal = sol.reshape((sol.shape[0], ))

    solexactreal = solexact.reshape((solexact.shape[0], ))
    
    solerr = solreal - solexactreal
    
   
    
    return max(abs(solerr)),wavenumber


if __name__ == '__main__':
    
    def aff(prods,order=True):
        
        affichage=[run_exercise_solution_helmholtz_dddd(prod,order=order) for prod in prods]
        
        max_err=[numpy.log(i[0]) for i in affichage]
        h_used=[numpy.log(i[1]) for i in affichage]
        
    
        regression=scipy.stats.linregress(h_used,max_err)
        affichage_reg=[regression[1]+i*regression[0] for i in h_used]
        import matplotlib.pyplot as plt
    
        plt.plot(h_used,max_err,label="real line")
        stri=f'slope ={int(regression[0]*100)/100}'
        plt.plot(h_used,affichage_reg,label="linear regression " +stri)
        
        stri2_ordered=f'plot for ordered meshs of wavenumbers between {prods[0]}pi and {prods[-1]}pi'
        stri2_unordered=f'plot for unordered mesh of wavenumbers between {prods[0]}pi and {prods[-1]}pi'
        stri2=stri2_unordered
        plt.xlabel("log(k)")
        plt.ylabel("log(err)")
        if order:
            stri2=stri2_ordered
        plt.title(stri2)
        plt.legend()
        plt.show()
        
    aff(numpy.linspace(0.1,3,20),order=False) 

    print('End.')