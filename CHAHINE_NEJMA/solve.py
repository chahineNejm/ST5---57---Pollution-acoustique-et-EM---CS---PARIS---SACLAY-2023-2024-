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
from creation_peigne import *
# MRG packages
import zsolutions4students as zsolutions
import solutions


def resolve_eq(p_elem2nodes, elem2nodes, node_coords,f_unassembled,values_at_nodes_on_boundary,frequency=2,affichage=True,D3=False,eigens=False):
    
    wavenumber=2*np.pi/frequency

    nelems=p_elem2nodes.shape[0]-1

    totalboundary=noeud_contour(node_coords, p_elem2nodes, elem2nodes) 
    boundary_left=[]
    for i in totalboundary:
        if node_coords[i][0]==0:
            boundary_left.append(i)
    
    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128) # stiffness element
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128) # mass element
    
    K, M, F = zsolutions._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M 
    B = F
    A, B = zsolutions._set_dirichlet_condition(boundary_left, values_at_nodes_on_boundary, A, B)
    sol = scipy.linalg.solve(A, B)
    solreal = sol.reshape((sol.shape[0], ))
    
   # we add in here options to visualise different plots
    if affichage:
        _ = solutions._plot_contourf_1(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
        if D3==True:
            _ = zsolutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
    if eigens:
        eigenvals=np.linalg.eig(A)[0]
        matplotlib.pyplot.scatter(numpy.real(eigenvals),numpy.imag(eigenvals))
        matplotlib.pyplot.show()
        
    return A,solreal


# we calculate the existence surface based on the formula given in the lecture
def calcule_surface_existence(solreal):
    vect=np.abs(solreal)
    norm=np.sqrt(sum(vect**2))
    S=sum(vect**4)
    return norm/S

# we calculate the energy dissipation based on the formula given in the lecture
def calcule_energie_dissipation(solreal,node_coords, p_elem2nodes, elem2nodes):
    total_boundary=noeud_contour(node_coords, p_elem2nodes, elem2nodes)
    vect=np.abs(solreal)
    norm=np.sqrt(sum(vect**2))
    W=sum(vect[total_boundary]**2)
    return W/norm

    



if __name__=="__main__":
    ### this part utilizes the previous functions to generate eigen values corresponding to the void and an absorbing substance

    p_elem2nodes, elem2nodes, node_coords = creation_maillage(hauteur=20, lar_moyenne=20, amplitude_max=1, amplitude_min=0, epaisseur=4, type='triangle', order=False)
    affichage(p_elem2nodes, elem2nodes, node_coords, barycentre=False)

    nnodes = node_coords.shape[0]

    values_at_nodes_on_boundary = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    f_unassembled = numpy.ones((nnodes, 1), dtype=numpy.complex128)

    # Solve the equation and obtain eigenvalues
    A, _ = resolve_eq(p_elem2nodes, elem2nodes, node_coords, f_unassembled, values_at_nodes_on_boundary, D3=True, eigens=False, frequency=0.5, affichage=False)
    eig_Val, eig_Vec = scipy.linalg.eig(A)

    # Plot eigenvalues as blue stars
    plt.scatter(numpy.imag(eig_Val), numpy.real(eig_Val), marker="*", c='b', label='Real wavenumber')

        # Now, modify the frequency to a complex number
    A, solreal = resolve_eq(p_elem2nodes, elem2nodes, node_coords, f_unassembled, values_at_nodes_on_boundary, D3=True, eigens=False, frequency=0.5*complex(1,-1), affichage=False)
    eig_Val, eig_Vec = scipy.linalg.eig(A)
    
    # Plot the eigenvalues as red circles
    plt.scatter(numpy.imag(eig_Val), numpy.real(eig_Val), marker="o", c='r', label='Complex wavenumber')

    # Show the legend and the plot
    plt.legend()
    plt.xlabel('Imaginary')
    plt.ylabel('Real')
    plt.grid(True)
    plt.show()

    
    

