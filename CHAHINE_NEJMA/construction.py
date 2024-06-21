# -*- coding: utf-8 -*-
"""
.. warning:: The explanations of the functions in this file and the details of
the programming methodology have been given during the lectures.
"""


# Python packages
import matplotlib.pyplot
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import numpy as np
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
from quality_analysis import *

# MRG packages
import solutions


def add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid_coords):  
       
    node_coords=np.concatenate((node_coords,nodeid_coords.reshape(1,3)))
    
    return node_coords, p_elem2nodes, elem2nodes



def add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid2nodes): 

    # Mettre à jour p_elem2nodes pour inclure le nouvel élément
    p_elem2nodes = np.append(p_elem2nodes, p_elem2nodes[-1] + len(elemid2nodes))

    # Concaténer elemid2nodes à elem2nodes
    elem2nodes = np.concatenate((elem2nodes, elemid2nodes))

    return node_coords, p_elem2nodes, elem2nodes


def remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid): 
    # Supprimer les noeuds de l'élément de elem2nodes
    nodes_to_remove = range(p_elem2nodes[elemid],p_elem2nodes[elemid+1])
    elem2nodes = np.delete(elem2nodes,nodes_to_remove)

    # Mettre à jour p_elem2nodes
    interval = p_elem2nodes[elemid+1] - p_elem2nodes[elemid]
    p_elem2nodes[elemid+1:] -= interval
    p_elem2nodes=np.delete(p_elem2nodes,elemid)

    return node_coords, p_elem2nodes, elem2nodes


def remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid):
     
    ensemble = []

    # Step 1: Find elements that contain the node to be removed
    for i in range(len(p_elem2nodes) - 1):
        if nodeid in elem2nodes[p_elem2nodes[i]:p_elem2nodes[i + 1]]:
            ensemble.append(i)

    # Step 2: Remove elements containing the node and update data structures
    ensemble.reverse()
    for i in ensemble:
        node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, i)

    # Step 3: Remove the node from node_coords
    node_coords = np.delete(node_coords, nodeid, axis=0)

    # Step 4: Update elem2nodes to account for the removed node
    for i in range(len(elem2nodes)):
        if elem2nodes[i] > nodeid:
            elem2nodes[i] -= 1

    return node_coords, p_elem2nodes, elem2nodes

   

def build_node2elems(p_elem2nodes, elem2nodes):   
    
        # elem2nodes connectivity matrix
        e2n_coef = numpy.ones(len(elem2nodes), dtype=numpy.int64)
        e2n_mtx = scipy.sparse.csr_matrix((e2n_coef, elem2nodes, p_elem2nodes))
        # node2elems connectivity matrix
        n2e_mtx = e2n_mtx.transpose()
        n2e_mtx = n2e_mtx.tocsr()
        # output
        p_node2elems = n2e_mtx.indptr
        node2elems = n2e_mtx.indices

        return p_node2elems, node2elems  
    
def noeud_contour(node_coords, p_elem2nodes, elem2nodes):
        
    # Step 1: Build node-to-elements mapping
    p_node2elems, _ = build_node2elems(p_elem2nodes, elem2nodes)

    # Step 2: Initialize the contour list
    contour = []

    # Determine the number of elements each node is expected to have
    number_elems_tohave = 6

    # Adjust the expected number of elements based on element type (4 in some cases)
    if p_elem2nodes[1] - p_elem2nodes[0] == 4:
        number_elems_tohave = 4

    # Step 3: Identify nodes that don't have the expected number of elements
    for i in range(len(p_node2elems) - 1):
        if p_node2elems[i + 1] - p_node2elems[i] < number_elems_tohave:
            contour.append(i)

    return contour


    

def compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes):

    spacedim = node_coords.shape[1]

    nelems = p_elem2nodes.shape[0] - 1

    # Initialize an array to store element coordinates
    elem_coords = np.zeros((nelems, spacedim), dtype=np.float64)

    # Calculate the barycenter for each element
    for i in range(len(p_elem2nodes) - 1):
        # Get the nodes that define the current element
        nodes = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i + 1]]

        # Calculate the barycenter as the average of node coordinates
        elem_coords[i, :] = np.average(node_coords[nodes, :], axis=0)

    return elem_coords




def run_exercise_a():
    """Generate grid with quadrangles.
    """
    
    # -- generate grid with quadrangles
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 10, 10
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    # .. todo:: Modify the line below to call to generate a grid with quadrangles
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = my_set_quadmesh(...)
    # .. note:: If you do not succeed, uncomment the following line to access the solution
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    
    

    barycentre=compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes)
    
    
    node_coords, p_elem2nodes, elem2nodes=remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, 5)

    # -- plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='yellow')
    
    
    for i in range(len(barycentre)):
        solutions._plot_node(p_elem2nodes, elem2nodes, barycentre , i , color='red', marker='o')
    
    
        
    elem = 70
        
    solutions._plot_elem(p_elem2nodes, elem2nodes, node_coords, elem, color='orange')
    matplotlib.pyplot.show()
    
    
    
    return


def run_exercise_b():
    """Generate grid with triangles.
    """
    # -- generate grid with triangles
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 10, 10
    nelems = nelemsx * nelemsy * 2
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    nodes_on_boundary = solutions._set_trimesh_boundary(nelemsx, nelemsy)

    barycentre=compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes)
    
    
    # -- plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='yellow')
    
    for i in range(len(barycentre)):
        solutions._plot_node(p_elem2nodes, elem2nodes, barycentre , i , color='red', marker='o')
    
    elem = 70
    solutions._plot_elem(p_elem2nodes, elem2nodes, node_coords, elem, color='orange')
    matplotlib.pyplot.show()
    
    s=build_node2elems(p_elem2nodes, elem2nodes)


    return


def run_exercise_c():
    pass


def run_exercise_d():
    pass


if __name__ == '__main__':

    run_exercise_a()
    #run_exercise_b()
    run_exercise_c()
    run_exercise_d()
    print('End.')
