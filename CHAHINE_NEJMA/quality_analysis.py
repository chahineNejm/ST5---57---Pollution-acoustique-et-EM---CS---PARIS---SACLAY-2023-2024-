import matplotlib.pyplot as plt
import numpy as np



# Function to calculate the distance between two points represented by their coordinates
def distance_nodes(node_coord1, node_coord2):
    # Calculate the vector distance between the two points
    vect_distance = node_coord2 - node_coord1
   
    return np.sqrt(np.dot(vect_distance, vect_distance))

# Function to compute properties of a triangle defined by its three vertices
def properties_triangle(coord1, coord2, coord3):
    
    # Calculate the lengths of the three sides of the triangle
    AB = distance_nodes(coord1, coord2)
    BC = distance_nodes(coord3, coord2)
    CA = distance_nodes(coord1, coord3)
    
    # Compute the semiperimeter (half of the perimeter)
    s = (AB + BC + CA) / 2
    
    # Use Heron's formula to calculate the area of the triangle
    area = np.sqrt(s * (s - AB) * (s - BC) * (s - CA))
    
    
    properties_dict = {
        "area": area,            # Area of the triangle
        "perimeter": s * 2,      # Perimeter of the triangle
        "hmax": max(AB, BC, CA)  # Maximum side length 
    }
    
    
    return properties_dict

def compute_aspect_ratio_of_element(node_coords, p_elem2nodes, elem2nodes, type="triangle"):

    # Initialize an array to store the aspect ratios
    aspect_ratio = np.zeros(len(p_elem2nodes) - 1, dtype=np.float64)

    if type == "triangle":
        # For triangles, calculate aspect ratio based on formula
        alpha = np.sqrt(3) / 6
        for elem in range(len(p_elem2nodes) - 1):
            nodes = elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem + 1]]
            prop = properties_triangle(node_coords[nodes[0]], node_coords[nodes[1]], node_coords[nodes[2]])
            rho = np.sqrt(2 * prop["area"] / prop["perimeter"])
            aspect_ratio[elem] = prop["hmax"] * alpha / rho

    elif type == "quad":
        # For quads, calculate aspect ratio using the Q formula
        for elem in range(len(p_elem2nodes) - 1):
            Q = 1
            nodes = elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem + 1]]
            for node1 in range(len(nodes)):
                Q -= np.abs(np.dot(node_coords[nodes[node1]] - node_coords[nodes[(node1 + 1) % 4]],
                                 node_coords[nodes[(node1 + 1) % 4]] - node_coords[nodes[(node1 + 2) % 4]])) / (
                         4 * np.linalg.norm(node_coords[nodes[node1]] - node_coords[nodes[(node1 + 1) % 4]]) *
                         np.linalg.norm(node_coords[nodes[(node1 + 2) % 4]] - node_coords[nodes[(node1 + 1) % 4]]))
            aspect_ratio[elem] = Q

    return aspect_ratio
 
def compute_edge_length_factor_of_element(node_coords, p_elem2nodes, elem2nodes):
   
    # Initialize an array to store the edge length factors
    edge_elem = np.zeros(len(p_elem2nodes) - 1, dtype=np.float64)

    for elem in range(len(p_elem2nodes) - 1):
        contour = 0
        hmin = np.inf
        nodes = elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem + 1]]
        number_of_edges = len(nodes)

        for node1 in range(len(nodes)):
            # Calculate the length of the edge between two consecutive nodes
            d = distance_nodes(node_coords[nodes[node1]], node_coords[nodes[(node1 + 1) % number_of_edges]])
            contour += d
            # Track the minimum edge length
            hmin = min(hmin, d)

        # Calculate the edge length factor and store it in the array
        edge_elem[elem] = hmin / (contour / number_of_edges)

    return edge_elem

           

def analysis(p_elem2nodes, elem2nodes, node_coords, element_type="triangle"):
    aspect_ratio = compute_aspect_ratio_of_element(node_coords, p_elem2nodes, elem2nodes, element_type)
    edge_length = compute_edge_length_factor_of_element(node_coords, p_elem2nodes, elem2nodes)

    # Create subplots for aspect ratio and edge length factor
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)) 

    # Plot aspect ratio histogram
    axes[0].hist(aspect_ratio, bins=20, color='blue', alpha=0.7)
    axes[0].set_title("Aspect Ratio")
    axes[0].set_xlabel("Aspect Ratio")
    axes[0].set_ylabel("Frequency")

    # Plot edge length factor histogram
    axes[1].hist(edge_length, bins=20, color='green', alpha=0.7)
    axes[1].set_title("Edge Length Factor")
    axes[1].set_xlabel("Edge Length Factor")
    axes[1].set_ylabel("Frequency")
    
    # Show the subplots
    plt.tight_layout()
    plt.show()





 