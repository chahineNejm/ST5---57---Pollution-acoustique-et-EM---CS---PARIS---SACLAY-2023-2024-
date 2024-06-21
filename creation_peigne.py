from construction import *
import solutions
from quality_analysis import *
import numpy as np

#this functions transforms square elements into triangular elements
def split_quad_tri(p_elem2nodes, elem2nodes, orientation='right'):

    # Initialize new arrays for triangular elements
    p_elem2nodes_tri = np.array(range(len(p_elem2nodes) * 2 - 1)) * 3
    elem2nodes_tri = np.array([0])

    for elem in range(len(p_elem2nodes) - 1):
        nodes = elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem + 1]]

        # Determine the order of nodes for the triangular split based on orientation
        if orientation == 'right':
            ordered_nodes = np.array([nodes[0], nodes[1], nodes[2], nodes[0], nodes[2], nodes[3]])
        elif orientation == "left":
            ordered_nodes = np.array([nodes[0], nodes[1], nodes[3], nodes[1], nodes[2], nodes[3]])

        # Append the ordered nodes to the elem2nodes_tri array
        elem2nodes_tri = np.append(elem2nodes_tri, ordered_nodes)

    # Remove the initial placeholder value (0) in elem2nodes_tri
    elem2nodes_tri = elem2nodes_tri[1:]

    return p_elem2nodes_tri, elem2nodes_tri

    
    
      
def affichage(p_elem2nodes, elem2nodes, node_coords,barycentre=False,nodes=False):
    # -- plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='blue')

    if barycentre:
        bary=compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes)
        for i in range(len(bary)):
            solutions._plot_node(p_elem2nodes, elem2nodes, bary , i , color='red', marker='o')

    if nodes:
        for i in range(len(node_coords)):
            solutions._plot_node(p_elem2nodes, elem2nodes, node_coords , i , color='green', marker='o')
            
    
    matplotlib.pyplot.show()
 
# this function determines nodes that are not connected to any element
def lonely_node(node_coords, p_elem2nodes, elem2nodes):
    lonely=[]  
    for i in range(len(node_coords)):
        if i not in elem2nodes:
            lonely.append(i)

    return lonely

#this function moves a node with a percentage relative to distance an according to the direction of vector 
def move_node(percent,node_id,vector,node_coords,dist): 
    
    normalized_vector=vector/np.linalg.norm(vector)
    node_coords[node_id]=node_coords[node_id]+normalized_vector*percent*dist
    return node_coords

#this function moves nodes randomly both in distance and distance
def disturbed_nodes(node_coords,p_elem2nodes,elem2nodes):
    all_nodes=set(elem2nodes)
    contour=set(noeud_contour(node_coords,p_elem2nodes,elem2nodes))
    dist=distance_nodes(node_coords[0],node_coords[1])
    
    interior_nodes=all_nodes.difference(contour)
    for node_id in interior_nodes:
        vector=np.random.uniform(low=0,high=1,size=2)
        vector=np.append(vector,0)
        percent=np.random.randint(50)/100
        node_coords=move_node(percent,node_id,vector,node_coords,dist)
    
    return node_coords,p_elem2nodes,elem2nodes
    
        

    
    

def creation_maillage(hauteur=10, lar_moyenne=10, amplitude_max=8, amplitude_min=2, epaisseur=1, type='triangle', order=False):
    # Define the mesh dimensions and parameters
    xmin, ymin, xmax, ymax = 0.0, 0.0, 0.2 * lar_moyenne, 0.1 * hauteur
    nelemsx, nelemsy = lar_moyenne * 2, hauteur
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)

    # Generate the initial quad mesh and get element data
    node_coords, _ , p_elem2nodes, elem2nodes = solutions._set_quadmesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)

    # Initialize a list to store elements to be removed
    elements_to_remove = []

    # Loop through the mesh to identify elements to be removed
    for lin in range(0, nelemsy):
        if lin % (2 * epaisseur) == 0:
            # Determine the number of elements to remove based on random amplitude generating peaks and valeys
            number = np.random.randint(low=amplitude_min, high=amplitude_max) * (1 - int(lin == (nelemsy - epaisseur) and (nelemsy // epaisseur) % 2 == 1))
        for col in range(0, nelemsx):
            if col >= lar_moyenne - number and lin % (2 * epaisseur) < epaisseur:
                elements_to_remove.append((lin, col))
            if col >= lar_moyenne + number and lin % (2 * epaisseur) >= epaisseur:
                elements_to_remove.append((lin, col))

    # Reverse the list of elements to be removed to start removing elements with highest id
    elements_to_remove.reverse()

    # Remove the identified elements from the mesh
    for element in elements_to_remove:
        if element[0] % 2 == 0:
            _, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, element[0] * nelemsx + element[1])
        else:
            _, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, (element[0]) * nelemsx + element[1])

    # Identify and remove "lonely" nodes from the mesh
    lonely = sorted(lonely_node(node_coords, p_elem2nodes, elem2nodes), reverse=True)
    
    
    
    
    for nodeid in lonely:
        
        node_coords, p_elem2nodes, elem2nodes = remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid)
        
    lonely = sorted(lonely_node(node_coords, p_elem2nodes, elem2nodes), reverse=True)

    # Optionally, apply node perturbation if 'order' is set to False
    if not order:
        node_coords, _, _ = disturbed_nodes(node_coords, p_elem2nodes, elem2nodes)

    # If 'type' is 'triangle', split quadrilateral elements into triangles
    if type == 'triangle':
        p_elem2nodes, elem2nodes = split_quad_tri(p_elem2nodes, elem2nodes, orientation='right')
        
    
    return p_elem2nodes, elem2nodes, node_coords






if __name__=="__main__":
    
    type="triangle"  ## triangle ou quad
    order=False # bool
    
    p_elem2nodes,elem2nodes, node_coords=creation_maillage(hauteur=8,lar_moyenne=2,amplitude_max=1,
                                                           amplitude_min=0,epaisseur=3,type=type,order= order) 
    
    affichage(p_elem2nodes,elem2nodes,node_coords,barycentre=True,nodes=True)
    
    #if not order:
        #analysis(p_elem2nodes, elem2nodes,node_coords,type)
    affichage()
     
    print("END")